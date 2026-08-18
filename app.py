from flask import Flask, render_template, request, redirect
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
import matplotlib
import numpy as np
import shap
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
import csv
from flask import Response


app = Flask(__name__)

model = joblib.load("models/logistic_model.pkl")
explainer = shap.LinearExplainer(model, model.coef_)
def create_feature_importance():

    feature_names = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges"
    ]

    coefficients = model.coef_[0]

    feature_data = list(zip(feature_names, coefficients))

    feature_data.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = feature_data[:10]

    names = [x[0] for x in top_features]
    values = [x[1] for x in top_features]

    plt.figure(figsize=(8,5))

    plt.barh(
        names[::-1],
        values[::-1]
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.title("Top 10 Feature Impact")
    plt.xlabel("Coefficient Impact")

    plt.tight_layout()

    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig(
        "static/feature_importance.png",
        dpi=150
    )

    plt.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():

    user_data = {}

    user_data["gender"] = int(request.form["gender"])
    user_data["SeniorCitizen"] = int(request.form["SeniorCitizen"])
    user_data["Partner"] = int(request.form["Partner"])
    user_data["Dependents"] = int(request.form["Dependents"])
    user_data["tenure"] = int(request.form["tenure"])
    user_data["PhoneService"] = int(request.form["PhoneService"])
    user_data["MultipleLines"] = int(request.form["MultipleLines"])
    user_data["InternetService"] = int(request.form["InternetService"])
    user_data["OnlineSecurity"] = int(request.form["OnlineSecurity"])
    user_data["OnlineBackup"] = int(request.form["OnlineBackup"])
    user_data["DeviceProtection"] = int(request.form["DeviceProtection"])
    user_data["TechSupport"] = int(request.form["TechSupport"])
    user_data["StreamingTV"] = int(request.form["StreamingTV"])
    user_data["StreamingMovies"] = int(request.form["StreamingMovies"])
    user_data["Contract"] = int(request.form["Contract"])
    user_data["PaperlessBilling"] = int(request.form["PaperlessBilling"])
    user_data["PaymentMethod"] = int(request.form["PaymentMethod"])
    user_data["MonthlyCharges"] = float(request.form["MonthlyCharges"])
    user_data["TotalCharges"] = float(request.form["TotalCharges"])

    df = pd.DataFrame([user_data])

    result = model.predict(df)
    probability = model.predict_proba(df)

    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    feature_names = df.columns.tolist()

    feature_impact = list(
        zip(feature_names, shap_values[0])
    )

    feature_impact.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_shap_features = feature_impact[:5]

    #print("SHAP:", top_shap_features)
    
    if result[0] == 1:

        confidence = round(probability[0][1] * 100, 2)
        output = "Customer Will Churn"
        risk = "High 🔴"
        color = "#dc3545"

        recommendation = [
            "Contact the customer immediately.",
            "Offer a discount or special plan.",
            "Suggest a yearly contract.",
            "Provide better customer support."
        ]

    else:

        confidence = round(probability[0][0] * 100, 2)
        output = "Customer Will Not Churn"
        risk = "Low 🟢"
        color = "#28a745"

        recommendation = [
            "Customer is likely to stay.",
            "Continue providing good service.",
            "Offer loyalty rewards.",
            "Maintain customer satisfaction."
        ]
    #print("Prediction:", output)
    #print("Classes:", model.classes_)
    conn = sqlite3.connect("churn.db")
    cursor = conn.cursor()

    current_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    cursor.execute("""
    INSERT INTO prediction_history(prediction, confidence, risk, date_time)
    VALUES (?, ?, ?, ?)
    """, (output, confidence, risk, current_time))

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        prediction=output,
        confidence=confidence,
        risk=risk,
        color=color,
        recommendation=recommendation,
        top_shap_features=top_shap_features
    )
    
@app.route("/history")
def history():

    search = request.args.get("search")

    conn = sqlite3.connect("churn.db")
    cursor = conn.cursor()

    if search:

        cursor.execute("""
        SELECT *
        FROM prediction_history
        WHERE CAST(id AS TEXT) LIKE ?
        OR prediction LIKE ?
        OR risk LIKE ?
        OR date_time LIKE ?
        ORDER BY id DESC
        """, (
            "%" + search + "%",
            "%" + search + "%",
            "%" + search + "%",
            "%" + search + "%"
        ))

    else:

        cursor.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
        """)

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )

@app.route("/clear")
def clear():

    conn = sqlite3.connect("churn.db")

    cursor = conn.cursor()

    cursor.execute("DELETE FROM prediction_history")

    conn.commit()

    conn.close()

    return redirect("/history")


@app.route("/export_pdf")
def export_pdf():

    conn = sqlite3.connect("churn.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM prediction_history
    ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    pdf = SimpleDocTemplate("Prediction_History_Report.pdf")

    data = []

    data.append([
        "ID",
        "Prediction",
        "Confidence",
        "Risk",
        "Date & Time"
    ])

    for row in records:

        data.append([
            row[0],
            row[1],
            str(row[2]) + "%",
            row[3],
            row[4]
        ])

    table = Table(data)

    table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),colors.blue),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),1,colors.black),
        ("BACKGROUND",(0,1),(-1,-1),colors.beige),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("BOTTOMPADDING",(0,0),(-1,0),10)

    ]))

    pdf.build([table])

    return send_file(
        "Prediction_History_Report.pdf",
        as_attachment=True
    )


@app.route("/export_csv")
def export_csv():

    conn = sqlite3.connect("churn.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM prediction_history
    ORDER BY id DESC
    """)

    records = cursor.fetchall()
    conn.close()

    def generate():

        yield "ID,Prediction,Confidence,Risk,Date & Time\n"

        for row in records:
            yield f"{row[0]},{row[1]},{row[2]}%,{row[3]},{row[4]}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            "attachment;filename=Prediction_History.csv"
        }
    )
@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("churn.db")
    cursor = conn.cursor()

    # Total Predictions
    cursor.execute("SELECT COUNT(*) FROM prediction_history")
    total = cursor.fetchone()[0]

    # Churn Count
    cursor.execute("""
    SELECT COUNT(*)
    FROM prediction_history
    WHERE prediction='Customer Will Churn'
    """)
    churn = cursor.fetchone()[0]

    # Not Churn Count
    cursor.execute("""
    SELECT COUNT(*)
    FROM prediction_history
    WHERE prediction='Customer Will Not Churn'
    """)
    not_churn = cursor.fetchone()[0]

    # Average Confidence
    cursor.execute("""
    SELECT ROUND(AVG(confidence),2)
    FROM prediction_history
    """)
    avg_confidence = cursor.fetchone()[0]

    if avg_confidence is None:
        avg_confidence = 0

    # Highest Confidence Prediction
    cursor.execute("""
    SELECT prediction, confidence
    FROM prediction_history
    ORDER BY confidence DESC
    LIMIT 1
    """)

    top_prediction = cursor.fetchone()

    if top_prediction:
        top_prediction_name = top_prediction[0]
        top_prediction_confidence = top_prediction[1]
    else:
        top_prediction_name = "No Data"
        top_prediction_confidence = 0

    # Prediction Insights

    if total > 0:
        churn_rate = round((churn / total) * 100, 2)
        retention_rate = round((not_churn / total) * 100, 2)
    else:
        churn_rate = 0
        retention_rate = 0

    # Recent Predictions
    cursor.execute("""
    SELECT
        id,
        CASE
            WHEN prediction='Customer Will Churn' THEN 'Churn'
            ELSE 'Not Churn'
        END AS prediction,
        confidence
    FROM prediction_history
    ORDER BY id DESC
    LIMIT 5
    """)

    recent_predictions = cursor.fetchall()

    # Pie Chart
    labels = ["Not Churn", "Churn"]
    sizes = [not_churn, churn]
    colors = ["green", "red"]

    plt.figure(figsize=(5,5))

    if total > 0:
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90
        )
    else:
        plt.text(
            0.5,
            0.5,
            "No Data Available",
            ha="center",
            va="center",
            fontsize=16
        )

    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/pie_chart.png")
    plt.close()

    # Bar Chart
    plt.figure(figsize=(5,4))

    plt.bar(
        ["Not Churn", "Churn"],
        [not_churn, churn],
        color=["green","red"]
    )

    plt.title("Customer Churn Count")
    plt.ylabel("Customers")

    plt.savefig("static/bar_chart.png")
    plt.close()

    # Trend Chart
    cursor.execute("""
    SELECT
    substr(date_time,1,10),
    COUNT(*)
    FROM prediction_history
    GROUP BY substr(date_time,1,10)
    ORDER BY substr(date_time,7,4),
    substr(date_time,4,2),
    substr(date_time,1,2)
    """)

    trend = cursor.fetchall()

    dates = []
    counts = []

    for row in trend:
        dates.append(row[0])
        counts.append(row[1])

    plt.figure(figsize=(7,4))

    plt.plot(
        dates,
        counts,
        marker="o",
        linewidth=3
    )

    plt.title("Prediction Trend")
    plt.xlabel("Date")
    plt.ylabel("Predictions")

    plt.grid(True)
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig("static/trend_chart.png")
    plt.close()
    create_feature_importance()
    conn.close()

    return render_template(
    "dashboard.html",
    total=total,
    churn=churn,
    not_churn=not_churn,
    avg_confidence=avg_confidence,
    recent_predictions=recent_predictions,
    top_prediction_name=top_prediction_name,
    top_prediction_confidence=top_prediction_confidence,
    churn_rate=churn_rate,
    retention_rate=retention_rate
)
    
if __name__ == "__main__": 
    app.run(debug=True)