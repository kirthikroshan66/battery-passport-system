import json
import math
import os
import sqlite3
import urllib.error
import urllib.request
from io import BytesIO, StringIO
from functools import wraps

import qrcode
from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "battery-passport-demo-secret")

DATABASE = os.path.join(BASE_DIR, "battery_passport_pro.db")
ADMIN_USERNAME = os.environ.get("BATTERY_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("BATTERY_ADMIN_PASSWORD", "admin123")


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS manufacturers (
                manufacturer_id INTEGER PRIMARY KEY,
                name TEXT,
                country TEXT
            );

            CREATE TABLE IF NOT EXISTS batteries (
                battery_id TEXT PRIMARY KEY,
                qr_id TEXT UNIQUE,
                manufacturer_id INTEGER,
                chemistry TEXT,
                capacity REAL,
                voltage REAL,
                manufacture_date TEXT,
                FOREIGN KEY (manufacturer_id)
                    REFERENCES manufacturers (manufacturer_id)
            );

            CREATE TABLE IF NOT EXISTS ownership_history (
                id INTEGER PRIMARY KEY,
                battery_id TEXT,
                owner_name TEXT,
                from_date TEXT,
                to_date TEXT,
                FOREIGN KEY (battery_id) REFERENCES batteries (battery_id)
            );

            CREATE TABLE IF NOT EXISTS service_history (
                id INTEGER PRIMARY KEY,
                battery_id TEXT,
                service_date TEXT,
                soh INTEGER,
                charge_cycles INTEGER,
                remarks TEXT,
                FOREIGN KEY (battery_id) REFERENCES batteries (battery_id)
            );

            CREATE TABLE IF NOT EXISTS recycling_records (
                id INTEGER PRIMARY KEY,
                battery_id TEXT,
                recycling_status TEXT,
                recycler TEXT,
                recovery_percent REAL,
                FOREIGN KEY (battery_id) REFERENCES batteries (battery_id)
            );

            CREATE TABLE IF NOT EXISTS blockchain_ledger (
                block_id INTEGER PRIMARY KEY,
                battery_id TEXT,
                event TEXT,
                timestamp TEXT,
                FOREIGN KEY (battery_id) REFERENCES batteries (battery_id)
            );

            CREATE INDEX IF NOT EXISTS idx_service_history_battery_date
                ON service_history (battery_id, service_date DESC);
            CREATE INDEX IF NOT EXISTS idx_ownership_history_battery_date
                ON ownership_history (battery_id, from_date DESC);
            CREATE INDEX IF NOT EXISTS idx_recycling_records_battery
                ON recycling_records (battery_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_blockchain_ledger_battery
                ON blockchain_ledger (battery_id, block_id DESC);
            """
        )
    finally:
        conn.close()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


def get_battery_detail(conn, battery_id):
    return conn.execute(
        """
        SELECT b.*, m.name AS manufacturer_name
        FROM batteries b
        LEFT JOIN manufacturers m ON m.manufacturer_id = b.manufacturer_id
        WHERE b.battery_id = ?
        """,
        (battery_id,),
    ).fetchone()


def get_battery_summary_rows(conn, search=None, manufacturer=None, chemistry=None):
    query = """
        SELECT
            b.*, m.name AS manufacturer_name,
            (
                SELECT sh.soh
                FROM service_history sh
                WHERE sh.battery_id = b.battery_id
                ORDER BY sh.service_date DESC
                LIMIT 1
            ) AS latest_soh,
            (
                SELECT sh.charge_cycles
                FROM service_history sh
                WHERE sh.battery_id = b.battery_id
                ORDER BY sh.service_date DESC
                LIMIT 1
            ) AS latest_cycles
        FROM batteries b
        LEFT JOIN manufacturers m ON m.manufacturer_id = b.manufacturer_id
    """
    clauses = []
    params = []

    if search:
        clauses.append(
            "(b.battery_id LIKE ? OR b.qr_id LIKE ? OR m.name LIKE ? OR b.chemistry LIKE ?)"
        )
        like = f"%{search}%"
        params.extend([like, like, like, like])

    if manufacturer:
        clauses.append("m.name = ?")
        params.append(manufacturer)

    if chemistry:
        clauses.append("b.chemistry = ?")
        params.append(chemistry)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY b.battery_id"
    return conn.execute(query, params).fetchall()


def predict_soh_from_service_history(conn, charge_cycles):
    if charge_cycles is None:
        return None

    rows = conn.execute(
        """
        SELECT soh, charge_cycles
        FROM service_history
        WHERE soh IS NOT NULL AND charge_cycles IS NOT NULL
        """
    ).fetchall()
    samples = []
    for row in rows:
        try:
            soh = float(row["soh"])
            cycles = float(row["charge_cycles"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(soh) and math.isfinite(cycles) and 0 <= soh <= 100 and cycles >= 0:
            samples.append((cycles, soh))

    if len(samples) < 3:
        return None

    try:
        cycles = float(charge_cycles)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cycles):
        return None

    cycle_values = [sample[0] for sample in samples]
    min_cycles = min(cycle_values)
    max_cycles = max(cycle_values)
    if min_cycles == max_cycles or not min_cycles <= cycles <= max_cycles:
        return None

    if min_cycles == max_cycles:
        return None

    model = LinearRegression()
    model.fit(
        [[sample_cycles] for sample_cycles, _ in samples],
        [soh for _, soh in samples],
    )
    estimate = float(model.predict([[cycles]])[0])
    return {
        "estimated_soh": round(min(100, max(0, estimate)), 1),
        "charge_cycles": cycles,
        "training_records": len(samples),
        "min_cycles": min_cycles,
        "max_cycles": max_cycles,
    }


def get_fleet_snapshot(conn):
    batteries = get_battery_summary_rows(conn)
    rows = []
    for battery in batteries:
        rows.append(
            {
                "battery_id": battery["battery_id"],
                "manufacturer": battery["manufacturer_name"] or "Unknown",
                "chemistry": battery["chemistry"] or "Unknown",
                "capacity": battery["capacity"],
                "voltage": battery["voltage"],
                "latest_soh": float(battery["latest_soh"] or 0),
                "charge_cycles": float(battery["latest_cycles"] or 0),
            }
        )
    return rows


def generate_ai_response(question, selected_battery=None):
    conn = get_db()
    fleet = get_fleet_snapshot(conn)
    conn.close()

    critical_assets = [b for b in fleet if b["latest_soh"] < 80]
    avg_soh = round(
        sum(b["latest_soh"] for b in fleet) / len(fleet), 1
    ) if fleet else 0
    avg_capacity = round(
        sum(float(b["capacity"]) for b in fleet) / len(fleet), 1
    ) if fleet else 0
    strongest = max(fleet, key=lambda item: item["latest_soh"], default=None)
    weakest = min(fleet, key=lambda item: item["latest_soh"], default=None)

    selected = None
    if selected_battery:
        selected = next(
            (b for b in fleet if b["battery_id"] == selected_battery),
            None,
        )

    context = [
        f"Fleet size: {len(fleet)} batteries.",
        f"Average capacity: {avg_capacity} kWh.",
        f"Average SoH: {avg_soh}%.",
        f"Critical assets below 80% SoH: {len(critical_assets)}.",
    ]
    if strongest:
        context.append(
            f"Best performer: {strongest['battery_id']} at {strongest['latest_soh']}% SoH."
        )
    if weakest:
        context.append(
            f"Most at-risk battery: {weakest['battery_id']} at {weakest['latest_soh']}% SoH."
        )
    if selected:
        context.append(
            f"Selected battery {selected['battery_id']}: chemistry {selected['chemistry']}, "
            f"capacity {selected['capacity']} kWh, SoH {selected['latest_soh']}%, "
            f"cycles {selected['charge_cycles']}."
        )

    prompt = (
        "You are a battery lifecycle advisor. Use the battery fleet facts below and answer the user's question "
        "in concise, practical terms.\n\n"
        + "\n".join(context)
        + f"\n\nUser question: {question or 'Summarize fleet health.'}"
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and question:
        payload = {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a battery passport AI assistant focused on asset health and maintenance recommendations.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        try:
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content.strip()
        except (urllib.error.URLError, ValueError, KeyError, TimeoutError):
            pass

    normalized = (question or "").lower()
    if not normalized:
        normalized = "overview"

    if selected:
        if "risk" in normalized or "health" in normalized or "soh" in normalized:
            if selected["latest_soh"] < 80:
                return (
                    f"{selected['battery_id']} is trending below the healthy threshold. "
                    f"Its SoH is {selected['latest_soh']}% and it should be prioritized for inspection, recorded maintenance, and a replacement review."
                )
            return (
                f"{selected['battery_id']} is in good condition with {selected['latest_soh']}% SoH. "
                f"Continue the current maintenance cadence and monitor charge cycles at {selected['charge_cycles']} for early degradation." 
            )
        return (
            f"{selected['battery_id']} is a {selected['chemistry']} unit from {selected['manufacturer']}. "
            f"It currently sits at {selected['latest_soh']}% SoH with {selected['charge_cycles']} cycles and a capacity of {selected['capacity']} kWh. "
            f"This is a solid operational profile for routine service planning."
        )

    if "attention" in normalized or "critical" in normalized or "risk" in normalized:
        if not critical_assets:
            return "No battery is currently below the 80% SoH attention threshold, so the fleet is in a healthy state overall."
        battery_names = ", ".join(b["battery_id"] for b in critical_assets[:3])
        return (
            f"The fleet has {len(critical_assets)} batteries under the attention threshold. "
            f"Priority review should start with {battery_names}. "
            "Inspect service records, confirm thermal behavior, and schedule preventative maintenance before capacity loss accelerates."
        )

    if "maintenance" in normalized or "service" in normalized or "recommend" in normalized:
        if critical_assets:
            return (
                f"Recommended next actions: inspect {critical_assets[0]['battery_id']} first, review recent cycling history, and allocate maintenance resources to the batteries below 80% SoH before scheduling new deployments."
            )
        return "No immediate maintenance escalation is required; continue regular telemetry reviews and keep the current preventive cycle."

    if "best" in normalized or "strongest" in normalized or "top" in normalized:
        if strongest:
            return (
                f"The strongest battery is {strongest['battery_id']}, with {strongest['latest_soh']}% SoH and {strongest['charge_cycles']} cycles. "
                "Use it as a benchmark for expected cycling performance and compare other assets against its operating pattern."
            )

    if "overview" in normalized or "fleet" in normalized or "summary" in normalized:
        if not fleet:
            return "There are no battery records in the database yet. Add fleet data to unlock AI insights."
        return (
            f"The fleet currently contains {len(fleet)} batteries with an average SoH of {avg_soh}% and average capacity of {avg_capacity} kWh. "
            f"{len(critical_assets)} assets need attention because they sit below 80% SoH. "
            "The overall profile is stable, but maintenance prioritization should focus on low-SoH assets first."
        )

    return (
        f"Based on the current fleet data, average SoH is {avg_soh}% and {len(critical_assets)} assets need attention. "
        "Continue monitoring battery health trends and prioritize the lowest-SoH units for inspection and proactive maintenance."
    )


@app.route("/")
def home():
    conn = get_db()
    search = request.args.get("q", "").strip()
    manufacturer = request.args.get("manufacturer", "").strip()
    chemistry = request.args.get("chemistry", "").strip()

    batteries = get_battery_summary_rows(conn, search, manufacturer, chemistry)
    manufacturers = conn.execute(
        "SELECT DISTINCT name FROM manufacturers WHERE name IS NOT NULL ORDER BY name"
    ).fetchall()
    chemistries = conn.execute(
        "SELECT DISTINCT chemistry FROM batteries WHERE chemistry IS NOT NULL ORDER BY chemistry"
    ).fetchall()
    total_batteries = len(batteries)
    avg_capacity = round(
        sum(float(b["capacity"]) for b in batteries) / total_batteries, 1
    ) if total_batteries else 0
    avg_soh = round(
        sum(float(b["latest_soh"] or 0) for b in batteries) / total_batteries, 1
    ) if total_batteries else 0
    critical_assets = sum(1 for b in batteries if (b["latest_soh"] or 0) < 80)
    conn.close()

    return render_template(
        "index.html",
        batteries=batteries,
        manufacturers=manufacturers,
        chemistries=chemistries,
        search=search,
        manufacturer=manufacturer,
        chemistry=chemistry,
        total_batteries=total_batteries,
        avg_capacity=avg_capacity,
        avg_soh=avg_soh,
        critical_assets=critical_assets,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("admin_logged_in"):
        return redirect(url_for("admin"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            next_url = request.args.get("next") or url_for("admin")
            return redirect(next_url)
        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("home"))


@app.route("/admin")
@login_required
def admin():
    conn = get_db()
    batteries = get_battery_summary_rows(conn)
    conn.close()
    return render_template("admin.html", batteries=batteries)


@app.route("/ai", methods=["GET", "POST"])
@app.route("/assistant", methods=["GET", "POST"])
def ai_insights():
    conn = get_db()
    batteries = get_battery_summary_rows(conn)
    conn.close()

    selected_battery = request.form.get("battery_id", request.args.get("battery_id", "")).strip()
    question = request.form.get("question", request.args.get("question", "")).strip()
    answer = ""

    if request.method == "POST" and question:
        answer = generate_ai_response(question, selected_battery or None)
    elif request.args.get("question"):
        answer = generate_ai_response(question, selected_battery or None)
    elif not question:
        answer = generate_ai_response(
            "Give a concise fleet overview and highlight maintenance priorities.",
            selected_battery or None,
        )

    return render_template(
        "ai.html",
        batteries=batteries,
        selected_battery=selected_battery,
        question=question,
        answer=answer,
    )


@app.route("/battery/<battery_id>/ai", methods=["GET", "POST"])
def battery_ai(battery_id):
    conn = get_db()
    batteries = get_battery_summary_rows(conn)
    conn.close()

    question = request.form.get("question", request.args.get("question", "")).strip()
    if not question:
        question = f"Assess the health of battery {battery_id} and suggest next actions."

    answer = generate_ai_response(question, battery_id)
    return render_template(
        "ai.html",
        batteries=batteries,
        selected_battery=battery_id,
        question=question,
        answer=answer,
    )


@app.route("/battery/<battery_id>")
def battery_detail(battery_id):
    conn = get_db()
    battery = get_battery_detail(conn, battery_id)

    if battery is None:
        conn.close()
        abort(404)

    services = conn.execute(
        """
        SELECT service_date, soh, charge_cycles, remarks
        FROM service_history
        WHERE battery_id = ?
        ORDER BY service_date DESC
        """,
        (battery_id,),
    ).fetchall()

    ownership = conn.execute(
        """
        SELECT owner_name, from_date, to_date
        FROM ownership_history
        WHERE battery_id = ?
        ORDER BY from_date DESC
        LIMIT 1
        """,
        (battery_id,),
    ).fetchone()

    ledger = conn.execute(
        """
        SELECT event, timestamp
        FROM blockchain_ledger
        WHERE battery_id = ?
        ORDER BY block_id DESC
        LIMIT 5
        """,
        (battery_id,),
    ).fetchall()

    recycling = conn.execute(
        """
        SELECT recycling_status, recycler, recovery_percent
        FROM recycling_records
        WHERE battery_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (battery_id,),
    ).fetchone()
    soh_prediction = predict_soh_from_service_history(
        conn,
        services[0]["charge_cycles"] if services else None,
    )
    conn.close()

    return render_template(
        "battery.html",
        battery=battery,
        services=services,
        ownership=ownership,
        ledger=ledger,
        recycling=recycling,
        soh_prediction=soh_prediction,
    )


@app.route("/battery/<battery_id>/qr")
@login_required
def battery_qr(battery_id):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(f"Battery Passport: {battery_id}")
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png", download_name=f"{battery_id}_qr.png")


@app.route("/export/csv")
@login_required
def export_all_csv():
    conn = get_db()
    batteries = get_battery_summary_rows(conn)
    conn.close()

    buffer = StringIO()
    writer = __import__("csv").writer(buffer)
    writer.writerow([
        "battery_id",
        "qr_id",
        "manufacturer",
        "chemistry",
        "capacity_kwh",
        "voltage_v",
        "manufacture_date",
        "latest_soh",
    ])
    for battery in batteries:
        writer.writerow([
            battery["battery_id"],
            battery["qr_id"],
            battery["manufacturer_name"] or "Unknown",
            battery["chemistry"],
            battery["capacity"],
            battery["voltage"],
            battery["manufacture_date"],
            battery["latest_soh"],
        ])

    output = BytesIO(buffer.getvalue().encode("utf-8"))
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="battery_fleet_export.csv",
    )


@app.route("/export/csv/<battery_id>")
@login_required
def export_single_csv(battery_id):
    conn = get_db()
    battery = get_battery_detail(conn, battery_id)
    if battery is None:
        conn.close()
        abort(404)

    services = conn.execute(
        "SELECT service_date, soh, charge_cycles, remarks FROM service_history WHERE battery_id = ? ORDER BY service_date DESC",
        (battery_id,),
    ).fetchall()
    conn.close()

    buffer = StringIO()
    writer = __import__("csv").writer(buffer)
    writer.writerow(["battery_id", "qr_id", "manufacturer", "chemistry", "capacity_kwh", "voltage_v", "manufacture_date"])
    writer.writerow([
        battery["battery_id"],
        battery["qr_id"],
        battery["manufacturer_name"] or "Unknown",
        battery["chemistry"],
        battery["capacity"],
        battery["voltage"],
        battery["manufacture_date"],
    ])
    writer.writerow([])
    writer.writerow(["service_date", "soh", "charge_cycles", "remarks"])
    for service in services:
        writer.writerow([service["service_date"], service["soh"], service["charge_cycles"], service["remarks"]])

    output = BytesIO(buffer.getvalue().encode("utf-8"))
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{battery_id}_passport.csv",
    )


@app.route("/export/pdf/<battery_id>")
@login_required
def export_pdf(battery_id):
    conn = get_db()
    battery = get_battery_detail(conn, battery_id)
    if battery is None:
        conn.close()
        abort(404)

    services = conn.execute(
        "SELECT service_date, soh, charge_cycles, remarks FROM service_history WHERE battery_id = ? ORDER BY service_date DESC LIMIT 5",
        (battery_id,),
    ).fetchall()
    ownership = conn.execute(
        "SELECT owner_name, from_date, to_date FROM ownership_history WHERE battery_id = ? ORDER BY from_date DESC LIMIT 1",
        (battery_id,),
    ).fetchone()
    conn.close()

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setTitle(f"Battery Passport - {battery_id}")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 800, f"Battery Passport: {battery_id}")
    pdf.setFont("Helvetica", 11)
    y = 770
    details = [
        ("QR ID", battery["qr_id"]),
        ("Manufacturer", battery["manufacturer_name"] or "Unknown"),
        ("Chemistry", battery["chemistry"]),
        ("Capacity", f"{battery['capacity']} kWh"),
        ("Voltage", f"{battery['voltage']} V"),
        ("Manufacture Date", battery["manufacture_date"]),
        ("Owner", ownership["owner_name"] if ownership else "N/A"),
        ("Ownership Period", f"{ownership['from_date']} to {ownership['to_date']}" if ownership else "N/A"),
    ]
    for label, value in details:
        pdf.drawString(50, y, f"{label}: {value}")
        y -= 18

    y -= 18
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Recent service records")
    y -= 16
    pdf.setFont("Helvetica", 10)
    for service in services:
        line = f"{service['service_date']} | SoH {service['soh']}% | Cycles {service['charge_cycles']} | {service['remarks']}"
        if y < 80:
            pdf.showPage()
            y = 800
        pdf.drawString(50, y, line)
        y -= 16

    pdf.save()
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{battery_id}_passport.pdf",

    )


init_db()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)