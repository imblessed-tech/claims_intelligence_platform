# src/claims/data/synthesizer.py

import pandas as pd
import numpy as np
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ── ICD-10 Codes ──────────────────────────────────────────────────────────────
# ICD-10 (International Classification of Diseases, 10th revision)
# is the global standard for classifying medical diagnoses.
# Every insurance claim must have one. This is what your NLP model will predict.

ICD_CODES = {
    "diabetes":         ("E11", "Type 2 Diabetes Mellitus"),
    "copd":             ("J44", "Chronic Obstructive Pulmonary Disease"),
    "hypertension":     ("I10", "Essential Hypertension"),
    "wellness":         ("Z00", "General Health Examination"),
    "hyperlipidemia":   ("E78", "Disorders of Lipoprotein Metabolism"),
    "obesity":          ("E66", "Obesity"),
    "respiratory":      ("J06", "Acute Upper Respiratory Infection"),
    "ischemic_heart":   ("I25", "Chronic Ischaemic Heart Disease"),
    "anxiety":          ("F41", "Anxiety Disorder"),
    "gastro":           ("K21", "Gastro-Oesophageal Reflux Disease"),
    "uti":              ("N39", "Urinary Tract Infection"),
    "back_pain":        ("M54", "Dorsalgia / Chronic Back Pain"),
    "asthma":           ("J45", "Asthma"),
    "anaemia":          ("D50", "Iron-Deficiency Anaemia"),
    "depression":       ("F32", "Depressive Episode"),
    "malaria":          ("B54", "Unspecified Malaria"),
    "typhoid":          ("A01", "Typhoid Fever"),
    "skin":             ("L30", "Dermatitis"),
    "eye":              ("H52", "Refractive Error"),
    "dental":           ("K08", "Dental Disorder"),
}


# ── Note Templates ────────────────────────────────────────────────────────────
# Each entry is a tuple: (note_text, diagnosis_key)
# diagnosis_key maps to ICD_CODES above.
# {placeholders} are filled with real values from the data row.

TEMPLATES = {

    # ──────────────────────────────────────────────────────────────────────────
    # CATEGORY: high_bmi_smoker
    # Condition: Age 35+, BMI > 30, smoker = yes
    # Dominant diagnoses: COPD, ischaemic heart disease, hypertension
    # ──────────────────────────────────────────────────────────────────────────

    "high_bmi_smoker": [

        (
            "Patient is a {age}-year-old {sex} presenting with chronic obstructive "
            "pulmonary disease (COPD). History of heavy tobacco use, approximately "
            "{smoking_years} pack-years. Current BMI is {bmi}. Patient reports "
            "progressive shortness of breath on exertion and chronic productive cough. "
            "Oxygen saturation at {spo2}% on room air. Prescribed bronchodilators "
            "(Salbutamol inhaler) and referred for pulmonary rehabilitation. "
            "Advised smoking cessation counselling.",
            "copd"
        ),

        (
            "A {age}-year-old {sex} smoker presents with worsening breathlessness "
            "and recurrent chest infections over the past six months. BMI recorded "
            "at {bmi}. Spirometry confirms obstructive pattern consistent with COPD "
            "Grade {copd_grade}. Blood pressure {bp_systolic}/{bp_diastolic} mmHg. "
            "Tiotropium inhaler added to current regimen. Pulmonary function tests "
            "scheduled. Patient counselled extensively on smoking cessation.",
            "copd"
        ),

        (
            "Presenting complaint: exertional dyspnoea and morning cough productive "
            "of yellowish sputum. Patient is a {age}-year-old {sex}, current smoker "
            "with a {smoking_years}-year history. BMI {bmi}. SpO2 {spo2}% at rest. "
            "Auscultation reveals bilateral wheeze. Peak flow 58% of predicted. "
            "Diagnosis: acute exacerbation of COPD. Commenced on prednisolone 30mg "
            "for 5 days and amoxicillin for secondary infection. Follow-up in 2 weeks.",
            "copd"
        ),

        (
            "This {age}-year-old {sex} attends for review of ischaemic heart disease "
            "and ongoing tobacco dependence. Smokes {smoking_years} cigarettes per day. "
            "BMI {bmi}. Chest pain on exertion — CCS Class II angina. "
            "Resting ECG: T-wave inversion in lateral leads. "
            "Blood pressure {bp_systolic}/{bp_diastolic} mmHg. "
            "Current medications: Aspirin 75mg, Atorvastatin 80mg, Bisoprolol 5mg. "
            "GTN spray prescribed for acute episodes. Cardiology referral expedited. "
            "Patient strongly advised to cease smoking immediately.",
            "ischemic_heart"
        ),

        (
            "A {age}-year-old obese {sex} smoker presents with central chest tightness "
            "radiating to the left arm, lasting approximately 20 minutes and resolving "
            "with rest. BMI {bmi}. BP {bp_systolic}/{bp_diastolic} mmHg. HR {hr} bpm. "
            "12-lead ECG performed — ST changes noted in V3-V5. Troponin I pending. "
            "Admitted for monitoring. Loading dose Aspirin given. "
            "Cardiology team notified for urgent review.",
            "ischemic_heart"
        ),

        (
            "Routine review of a {age}-year-old {sex} with poorly controlled hypertension "
            "and active smoking. BMI {bmi}. BP today {bp_systolic}/{bp_diastolic} mmHg — "
            "significantly above target despite current Amlodipine 10mg. "
            "Fundoscopy reveals Grade II hypertensive retinopathy. "
            "Renal function slightly impaired — eGFR {egfr} ml/min. "
            "Ramipril 5mg added. Lifestyle counselling provided. "
            "Return in 4 weeks for BP review.",
            "hypertension"
        ),

        (
            "Patient, a {age}-year-old {sex}, presents with chronic smoker's cough "
            "and recurrent sinusitis. {smoking_years} pack-year history. BMI {bmi}. "
            "No significant spirometric obstruction at this visit — early COPD changes noted. "
            "Nasal corticosteroid spray commenced. "
            "Chest X-ray: hyperinflation consistent with early emphysema. "
            "Nicotine replacement therapy discussed and started. "
            "Advised annual spirometry and influenza vaccination.",
            "respiratory"
        ),
    ],


    # ──────────────────────────────────────────────────────────────────────────
    # CATEGORY: high_bmi_nonsmoker
    # Condition: BMI > 30, smoker = no
    # Dominant diagnoses: diabetes, hypertension, hyperlipidaemia, back pain
    # ──────────────────────────────────────────────────────────────────────────

    "high_bmi_nonsmoker": [

        (
            "Patient is a {age}-year-old {sex} presenting for diabetic review. "
            "BMI is {bmi}, weight {weight}kg. HbA1c recorded at {hba1c}%, "
            "indicating poorly controlled Type 2 Diabetes Mellitus. "
            "Fasting blood glucose {fbs} mmol/L. "
            "Blood pressure {bp_systolic}/{bp_diastolic} mmHg. "
            "Dietary modification recommended. Metformin dosage adjusted to 1000mg BD. "
            "Referred to dietician and diabetes educator.",
            "diabetes"
        ),

        (
            "A {age}-year-old obese {sex} with known Type 2 Diabetes and hypertension "
            "attends for quarterly review. BMI {bmi}. HbA1c {hba1c}% — sub-optimal. "
            "Currently on Metformin 500mg and Amlodipine 5mg. "
            "BP today {bp_systolic}/{bp_diastolic} mmHg — not at target. "
            "Lipid profile: elevated LDL at {ldl} mmol/L. "
            "Atorvastatin 20mg commenced. Weight loss programme discussed. "
            "Retinal screening and foot examination due — referral placed.",
            "diabetes"
        ),

        (
            "Follow-up consultation for a {age}-year-old {sex} with insulin-resistant "
            "Type 2 Diabetes. BMI {bmi}. Patient reports fatigue and polydipsia. "
            "HbA1c {hba1c}%. Random blood sugar {fbs} mmol/L. "
            "Urine dipstick: glucose 3+, protein trace. "
            "Renal function tests ordered. Empagliflozin added to regimen — "
            "dual benefit of glucose control and cardiovascular protection discussed. "
            "Patient educated on hypoglycaemia recognition and foot care.",
            "diabetes"
        ),

        (
            "A {age}-year-old {sex} presents with persistent headache and blurred "
            "vision for 3 days. BMI {bmi}. BP {bp_systolic}/{bp_diastolic} mmHg — "
            "severely elevated. No prior hypertension diagnosis. "
            "Urinalysis: protein 1+. ECG: LVH pattern. "
            "Admitted for observation and urgent antihypertensive initiation. "
            "Amlodipine 10mg and Ramipril 5mg commenced. "
            "Echocardiogram and 24-hour ambulatory BP monitoring requested.",
            "hypertension"
        ),

        (
            "Routine health assessment for a {age}-year-old obese {sex}. "
            "BMI {bmi}. Complains of low back pain for 3 months — worse on prolonged "
            "sitting. No neurological deficits on examination. "
            "Lumbar spine X-ray: mild degenerative changes at L4-L5. "
            "Diagnosed with mechanical low back pain secondary to obesity. "
            "Referred to physiotherapy. NSAIDs prescribed short term. "
            "Weight loss strongly advised as primary intervention.",
            "back_pain"
        ),

        (
            "A {age}-year-old {sex} with elevated BMI of {bmi} presents for lipid "
            "profile review. Fasting cholesterol {cholesterol} mmol/L, "
            "LDL {ldl} mmol/L, HDL {hdl} mmol/L, triglycerides {triglycerides} mmol/L. "
            "No current cardiovascular symptoms. BP {bp_systolic}/{bp_diastolic} mmHg. "
            "10-year cardiovascular risk calculated at {cv_risk}%. "
            "Rosuvastatin 10mg initiated. Mediterranean diet leaflet provided. "
            "Exercise of 150 minutes moderate intensity per week advised.",
            "hyperlipidemia"
        ),

        (
            "This {age}-year-old {sex} presents with reflux symptoms — heartburn and "
            "regurgitation, particularly after meals. BMI {bmi}. Symptoms worsened "
            "with weight gain over the past year. "
            "Abdominal examination: mild epigastric tenderness. "
            "Diagnosed with gastro-oesophageal reflux disease (GORD). "
            "Omeprazole 20mg OD prescribed. Advised weight reduction, elevation of "
            "head of bed, and avoidance of late meals.",
            "gastro"
        ),
    ],


    # ──────────────────────────────────────────────────────────────────────────
    # CATEGORY: young_healthy
    # Condition: Age < 35, BMI < 28, smoker = no
    # Dominant diagnoses: wellness, anxiety, UTI, asthma, eye, dental
    # ──────────────────────────────────────────────────────────────────────────

    "young_healthy": [

        (
            "Patient is a {age}-year-old {sex} presenting for annual wellness examination. "
            "No significant medical complaints. BMI {bmi} — within normal range. "
            "Blood pressure {bp_systolic}/{bp_diastolic} mmHg. "
            "Heart rate {hr} bpm, regular. All systems review unremarkable. "
            "Advised to maintain current lifestyle. Routine bloods requested. "
            "Next review in 12 months.",
            "wellness"
        ),

        (
            "Annual health check for a {age}-year-old {sex}. "
            "Patient reports good general health. No chronic conditions documented. "
            "BMI {bmi}. Non-smoker. Vitals within normal limits. "
            "BP {bp_systolic}/{bp_diastolic} mmHg. HR {hr} bpm. "
            "Immunisation history reviewed — up to date. "
            "No current medications. Preventive health counselling provided.",
            "wellness"
        ),

        (
            "A {age}-year-old {sex} presents with a 2-week history of generalised "
            "anxiety, poor sleep, and difficulty concentrating at work. "
            "No prior psychiatric history. BMI {bmi}. "
            "PHQ-9 score: {phq9}/27 (mild depression component). "
            "GAD-7 score: {gad7}/21 (moderate anxiety). "
            "Cognitive behavioural therapy referral made. "
            "Sleep hygiene advice provided. Reassured regarding physical health. "
            "Follow-up in 4 weeks.",
            "anxiety"
        ),

        (
            "A {age}-year-old {sex} attends with dysuria, increased urinary frequency, "
            "and mild suprapubic discomfort for 3 days. No fever or loin pain. "
            "BMI {bmi}. Urine dipstick: leucocytes 3+, nitrites positive. "
            "Midstream urine sent for culture. "
            "Diagnosed with uncomplicated lower urinary tract infection. "
            "Nitrofurantoin 100mg MR BD for 5 days prescribed. "
            "Advised increased fluid intake. Return if not improving in 48 hours.",
            "uti"
        ),

        (
            "A {age}-year-old {sex} with known asthma presents for routine review. "
            "Current symptoms: occasional wheeze and nocturnal cough, "
            "approximately 3 episodes per week. BMI {bmi}. "
            "PEFR today: {pefr}% of predicted. "
            "Inhaler technique assessed — satisfactory. "
            "Step-up to low-dose inhaled corticosteroid (Budesonide 200mcg BD) "
            "plus SABA as needed. Asthma action plan updated. "
            "Allergen avoidance discussed.",
            "asthma"
        ),

        (
            "A {age}-year-old {sex} presents with blurred vision and headaches "
            "when reading. No eye pain, discharge, or trauma. BMI {bmi}. "
            "Visual acuity: RE {va_right}/6, LE {va_left}/6 unaided. "
            "Refraction performed — mild myopia confirmed bilaterally. "
            "Spectacles prescribed. Advised annual review. "
            "Intraocular pressure within normal limits.",
            "eye"
        ),

        (
            "A {age}-year-old {sex} presents with toothache and sensitivity to cold "
            "in the lower left quadrant for one week. BMI {bmi}. "
            "Oral examination: carious lesion at LL6 with periapical tenderness. "
            "Periapical X-ray: early periapical abscess. "
            "Root canal treatment initiated. Amoxicillin 500mg TDS for 5 days prescribed. "
            "Patient advised on oral hygiene and regular dental review.",
            "dental"
        ),
    ],


    # ──────────────────────────────────────────────────────────────────────────
    # CATEGORY: elderly_hypertension
    # Condition: Age > 55
    # Dominant diagnoses: hypertension, ischaemic heart, diabetes, depression
    # ──────────────────────────────────────────────────────────────────────────

    "elderly_hypertension": [

        (
            "Patient is a {age}-year-old {sex} with longstanding hypertension "
            "and hyperlipidaemia. Current medications: Amlodipine 10mg, "
            "Atorvastatin 40mg, Aspirin 75mg. "
            "Blood pressure today {bp_systolic}/{bp_diastolic} mmHg — "
            "not at target. ECG shows normal sinus rhythm. "
            "Ramipril 5mg added to antihypertensive regimen. "
            "Referred to cardiology for further evaluation.",
            "hypertension"
        ),

        (
            "A {age}-year-old {sex} reviewed for hypertension management. "
            "BP {bp_systolic}/{bp_diastolic} mmHg. BMI {bmi}. "
            "Reports occasional headaches and dizziness. "
            "Renal function tests within normal limits. eGFR {egfr} ml/min. "
            "Antihypertensive medications optimised — Losartan dose increased. "
            "Salt restriction (<5g/day) and aerobic exercise advised. "
            "24-hour ambulatory BP monitoring arranged. Follow-up in 4 weeks.",
            "hypertension"
        ),

        (
            "A {age}-year-old {sex} presents with 3-month history of low mood, "
            "anhedonia, poor appetite, and disturbed sleep. "
            "Recently widowed. BMI {bmi}. PHQ-9 score: {phq9}/27 — moderate depression. "
            "No suicidal ideation expressed. BP {bp_systolic}/{bp_diastolic} mmHg. "
            "Sertraline 50mg OD commenced with plan to review in 4 weeks. "
            "Referred to counselling service. Safety-netting advice provided. "
            "GP to follow up closely.",
            "depression"
        ),

        (
            "This {age}-year-old {sex} with known ischaemic heart disease attends "
            "for 6-month cardiology review. CCS Class I angina — stable. "
            "BMI {bmi}. BP {bp_systolic}/{bp_diastolic} mmHg. HR {hr} bpm. "
            "Resting ECG: old inferior Q waves, no acute changes. "
            "Echocardiogram shows mildly impaired LV function — EF {ef}%. "
            "Current medications: Bisoprolol, Ramipril, Atorvastatin, Aspirin, GTN. "
            "Stress test scheduled. Lifestyle counselling reinforced.",
            "ischemic_heart"
        ),

        (
            "A {age}-year-old {sex} with a 12-year history of Type 2 Diabetes "
            "presents for annual review. BMI {bmi}. HbA1c {hba1c}%. "
            "Peripheral neuropathy confirmed — monofilament test reduced bilaterally. "
            "Fundoscopy: background diabetic retinopathy. "
            "ACR {acr} mg/mmol — microalbuminuria present. "
            "Intensification of diabetes management discussed. "
            "Podiatry and ophthalmology referrals placed. "
            "Patient educated on foot care, signs of hypoglycaemia.",
            "diabetes"
        ),

        (
            "Review of a {age}-year-old {sex} with multiple comorbidities: "
            "hypertension, hyperlipidaemia, and osteoarthritis of bilateral knees. "
            "BMI {bmi}. BP {bp_systolic}/{bp_diastolic} mmHg — controlled. "
            "Knee pain limiting mobility — WOMAC score {womac}/96. "
            "Current analgesia: Paracetamol 1g QDS. "
            "Topical Diclofenac added. Physiotherapy referral for knee strengthening. "
            "Weight loss discussed as key intervention to reduce joint load. "
            "Orthopaedic review if no improvement in 3 months.",
            "back_pain"
        ),

        (
            "A {age}-year-old {sex} presents with 4-day history of high-grade fever, "
            "rigors, headache, and myalgia. Returned from Kano 5 days ago. "
            "BMI {bmi}. Temperature {temp}°C. BP {bp_systolic}/{bp_diastolic} mmHg. "
            "Malaria RDT: positive for Plasmodium falciparum. "
            "Commenced on Artemether-Lumefantrine (Coartem). "
            "FBC: Hb {hb} g/dL — mild anaemia. Advised rest and adequate hydration. "
            "Review in 48 hours to confirm parasite clearance.",
            "malaria"
        ),
    ],


    # ──────────────────────────────────────────────────────────────────────────
    # CATEGORY: general
    # Condition: Everyone else — mixed presentations
    # Dominant diagnoses: respiratory, anaemia, typhoid, skin, wellness
    # ──────────────────────────────────────────────────────────────────────────

    "general": [

        (
            "Patient is a {age}-year-old {sex} presenting with general fatigue "
            "and mild upper respiratory symptoms for 5 days. "
            "Temperature {temp}°C. BP {bp_systolic}/{bp_diastolic} mmHg. BMI {bmi}. "
            "Throat mildly erythematous. Chest clear on auscultation. "
            "Diagnosed with acute upper respiratory tract infection — viral aetiology likely. "
            "Prescribed symptomatic treatment: paracetamol and steam inhalation. "
            "Advised rest and hydration. Return if symptoms worsen.",
            "respiratory"
        ),

        (
            "A {age}-year-old {sex} presents with a 2-week history of progressive "
            "generalised weakness and pallor. BMI {bmi}. Palpitations on exertion. "
            "Conjunctival pallor noted. FBC: Hb {hb} g/dL — moderate anaemia. "
            "MCV low at {mcv} fl — consistent with iron deficiency. "
            "Serum ferritin {ferritin} ng/mL — depleted. "
            "Oral ferrous sulphate 200mg TDS prescribed for 3 months. "
            "Dietary advice on iron-rich foods given. "
            "Source of blood loss to be investigated.",
            "anaemia"
        ),

        (
            "A {age}-year-old {sex} with persistent fever, abdominal discomfort, "
            "and constipation for 10 days. BMI {bmi}. Temperature {temp}°C. "
            "Tongue coated. Mild hepatosplenomegaly on palpation. "
            "Widal test: O antigen titre 1:160 — significant. "
            "Blood culture collected. Clinically consistent with typhoid fever. "
            "Commenced on Ciprofloxacin 500mg BD for 14 days. "
            "Isolation precautions advised. Nutritional support given.",
            "typhoid"
        ),

        (
            "This {age}-year-old {sex} presents with a pruritic rash on the "
            "trunk and upper limbs for 3 weeks. BMI {bmi}. No fever. "
            "Examination: erythematous papular rash with excoriations. "
            "No vesicles or pustules. History of atopic eczema in childhood. "
            "Diagnosed with atopic dermatitis — acute flare. "
            "Hydrocortisone 1% cream BD for 2 weeks. "
            "Emollient (aqueous cream) prescribed. Antihistamine at night for pruritus. "
            "Triggers discussed — soap, synthetic fabrics, stress.",
            "skin"
        ),

        (
            "A {age}-year-old {sex} attends for occupational health pre-employment screening. "
            "No presenting complaints. BMI {bmi}. "
            "BP {bp_systolic}/{bp_diastolic} mmHg. Visual acuity: 6/6 bilaterally. "
            "Audiometry: normal hearing thresholds. Urine dipstick: clear. "
            "FBC and metabolic panel requested. Hepatitis B serology: immune. "
            "Fit for employment with no restrictions noted. "
            "Advised on workplace health and safety practices.",
            "wellness"
        ),

        (
            "A {age}-year-old {sex} presents with palpitations and mild breathlessness "
            "for 2 days. BMI {bmi}. BP {bp_systolic}/{bp_diastolic} mmHg. "
            "HR {hr} bpm — irregular. "
            "ECG: atrial fibrillation with ventricular rate {hr} bpm. "
            "Thyroid function tests and echocardiogram requested. "
            "Rate control initiated with Bisoprolol 2.5mg. "
            "CHA2DS2-VASc score calculated — anticoagulation discussed. "
            "Cardiology referral placed. Patient advised not to drive until reviewed.",
            "ischemic_heart"
        ),

        (
            "Follow-up for a {age}-year-old {sex} recently treated for malaria "
            "two weeks ago. BMI {bmi}. Fever resolved on day 3 of treatment. "
            "Currently asymptomatic. Repeat malaria RDT: negative. "
            "FBC: Hb recovering at {hb} g/dL — up from nadir. "
            "Advised to continue haematinics for 4 weeks. "
            "Prevention counselled: insecticide-treated nets, "
            "indoor residual spraying, and chemoprophylaxis for future travel.",
            "malaria"
        ),
    ],
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def _get_template_key(row: pd.Series) -> str:
    """
    Decide which template category applies to this patient row.
    Order of conditions matters — check most specific first.
    """
    age = row["age"]
    bmi = row["bmi"]
    smoker = row["smoker"]

    if smoker == "yes" and bmi > 30:
        return "high_bmi_smoker"
    elif smoker == "no" and bmi > 30:
        return "high_bmi_nonsmoker"
    elif age < 35 and bmi < 28:
        return "young_healthy"
    elif age > 55:
        return "elderly_hypertension"
    else:
        return "general"


def _fill_template(template: str, row: pd.Series) -> str:
    """
    Fill all {placeholder} values in the template string.
    Uses numpy seeded RNG so the same row always gets the same values —
    this makes the dataset reproducible across runs.
    """
        # Ensure the seed is an integer for np.random.default_rng
    if isinstance(row.name, int):
        seed = row.name
    else:
        seed = sum(ord(c) for c in str(row.name))
    rng = np.random.default_rng(seed=seed)

    values = {
        # ── From the actual data row ─────────────────────────────────────────
        "age":            int(row["age"]),
        "sex":            "male" if row["sex"] == "male" else "female",
        "bmi":            round(float(row["bmi"]), 1),
        "weight":         round(float(row["bmi"]) * 1.75, 1),  # approximate kg

        # ── Smoking history ──────────────────────────────────────────────────
        "smoking_years":  int(rng.integers(5, 30)),
        "copd_grade":     str(rng.choice(["II", "III"])),

        # ── Respiratory ──────────────────────────────────────────────────────
        "spo2":           int(rng.integers(91, 97)),
        "pefr":           int(rng.integers(55, 80)),

        # ── Cardiovascular ───────────────────────────────────────────────────
        "bp_systolic":    int(rng.integers(118, 168)),
        "bp_diastolic":   int(rng.integers(74, 106)),
        "hr":             int(rng.integers(62, 96)),
        "ef":             int(rng.integers(40, 56)),   # ejection fraction %

        # ── Metabolic / diabetes ─────────────────────────────────────────────
        "hba1c":          round(float(rng.uniform(7.5, 11.5)), 1),
        "fbs":            round(float(rng.uniform(7.0, 14.0)), 1),  # mmol/L
        "egfr":           int(rng.integers(45, 90)),
        "acr":            int(rng.integers(3, 31)),    # albumin:creatinine ratio

        # ── Lipids ───────────────────────────────────────────────────────────
        "cholesterol":    round(float(rng.uniform(4.5, 8.1)), 1),
        "ldl":            round(float(rng.uniform(3.0, 6.6)), 1),
        "hdl":            round(float(rng.uniform(0.8, 1.9)), 1),
        "triglycerides":  round(float(rng.uniform(1.2, 4.6)), 1),
        "cv_risk":        int(rng.integers(8, 29)),    # 10-year CV risk %

        # ── Haematology ──────────────────────────────────────────────────────
        "hb":             round(float(rng.uniform(7.5, 13.6)), 1),  # haemoglobin g/dL
        "mcv":            int(rng.integers(62, 80)),   # mean corpuscular volume fl
        "ferritin":       int(rng.integers(3, 15)),    # ng/mL

        # ── General vitals ───────────────────────────────────────────────────
        "temp":           round(float(rng.uniform(36.5, 38.6)), 1),

        # ── Psychiatry scoring ───────────────────────────────────────────────
        "phq9":           int(rng.integers(5, 20)),    # depression score /27
        "gad7":           int(rng.integers(8, 18)),    # anxiety score /21

        # ── Ophthalmology ────────────────────────────────────────────────────
        "va_right":       int(rng.choice([6, 9, 12, 18])),   # visual acuity
        "va_left":        int(rng.choice([6, 9, 12, 18])),

        # ── Orthopaedics ─────────────────────────────────────────────────────
        "womac":          int(rng.integers(35, 73)),   # knee pain score /96
    }

    return template.format(**values)


# ── Main Synthesizer Class ────────────────────────────────────────────────────

class ClinicalNoteSynthesizer:
    """
    Generates synthetic clinical notes for each row of the insurance DataFrame.

    Each row is assigned a template category based on its clinical profile,
    then a random template variant is selected from that category.
    All randomness is seeded per row for reproducibility.

    Usage:
        synthesizer = ClinicalNoteSynthesizer()
        enriched_df = synthesizer.generate(df)
        synthesizer.save(enriched_df, output_path)
    """

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add clinical_note, icd_code, and icd_description columns to DataFrame.
        Returns a new DataFrame — does not modify the original.
        """
        logger.info(f"Generating clinical notes for {len(df)} claims...")

        notes = []
        icd_codes = []
        icd_descriptions = []
        template_categories = []

        for idx, row in df.iterrows():
            # Determine which template group this patient falls into
            template_key = _get_template_key(row)

            # Ensure the Hashable index is a valid seed type (int or str)
            if isinstance(idx, (int, str)):
                seed = idx
            else:
                seed = str(idx)
            rng = random.Random(seed)

            template_text, diagnosis_key = rng.choice(TEMPLATES[template_key])

            # Fill in the placeholder values
            note = _fill_template(template_text, row)

            # Look up the ICD code for this diagnosis
            icd_code, icd_desc = ICD_CODES[diagnosis_key]

            notes.append(note)
            icd_codes.append(icd_code)
            icd_descriptions.append(icd_desc)
            template_categories.append(template_key)

        # Build the enriched DataFrame
        df = df.copy()
        df["clinical_note"] = notes
        df["icd_code"] = icd_codes
        df["icd_description"] = icd_descriptions
        df["patient_category"] = template_categories  # useful for analysis

        # Log distribution summary
        category_counts = df["patient_category"].value_counts()
        icd_counts = df["icd_code"].value_counts()
        logger.info("Patient category distribution:")
        for cat, count in category_counts.items():
            logger.info(f"  {cat}: {count} ({count/len(df):.1%})")
        logger.info("Top ICD codes generated:")
        for code, count in icd_counts.head(8).items():
            desc = ICD_CODES.get(
                next((k for k, v in ICD_CODES.items() if v[0] == code), ""),
                (code, "")
            )[1]
            logger.info(f"  {code} ({desc}): {count}")

        logger.info(
            f"Clinical note generation complete. "
            f"Unique ICD codes: {df['icd_code'].nunique()}"
        )
        return df

    def save(self, df: pd.DataFrame, path: Path) -> None:
        """Save enriched DataFrame to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved enriched dataset to {path} ({len(df)} rows, {len(df.columns)} columns)")


# ── Convenience function ──────────────────────────────────────────────────────

def synthesize(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """One-liner convenience wrapper."""
    s = ClinicalNoteSynthesizer()
    enriched = s.generate(df)
    s.save(enriched, output_path)
    return enriched