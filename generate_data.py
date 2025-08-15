import csv
import random
from faker import Faker
import time

fake = Faker()

PROFILES = {
    # Cardiology
    "cardiac_hypertension": {
        "symptoms": ["chest pain", "shortness of breath", "dizziness", "high blood pressure reading", "headache"],
        "history": ["History of hypertension", "Family history of heart disease", "Smoker", "High cholesterol", "Poor diet"],
        "diagnosis_code": ["I10 - Essential (primary) hypertension", "I25.10 - Atherosclerotic heart disease"]
    },
    # Endocrinology
    "diabetes_type2": {
        "symptoms": ["increased thirst", "frequent urination", "unexplained weight loss", "fatigue", "blurred vision"],
        "history": ["Obesity (BMI > 30)", "Sedentary lifestyle", "Family history of diabetes", "Prediabetes diagnosis"],
        "diagnosis_code": ["E11.9 - Type 2 diabetes mellitus without complications", "E11.65 - Type 2 diabetes with hyperglycemia"]
    },
    "hypothyroidism": {
        "symptoms": ["fatigue", "weight gain", "cold intolerance", "dry skin", "constipation", "hair loss"],
        "history": ["Family history of thyroid disease", "Autoimmune conditions (e.g., Hashimoto's)", "Previous thyroid surgery"],
        "diagnosis_code": ["E03.9 - Hypothyroidism, unspecified"]
    },
    # Pulmonology / Respiratory
    "respiratory_asthma": {
        "symptoms": ["wheezing", "coughing", "chest tightness", "shortness of breath during exercise"],
        "history": ["Childhood asthma", "History of seasonal allergies", "Eczema", "Exposure to environmental triggers"],
        "diagnosis_code": ["J45.909 - Unspecified asthma, uncomplicated", "J45.40 - Moderate persistent asthma"]
    },
    "pneumonia": {
        "symptoms": ["cough with phlegm or pus", "fever", "chills", "difficulty breathing", "sharp chest pain when breathing"],
        "history": ["Recent cold or flu", "Weakened immune system", "Smoker", "Hospitalization"],
        "diagnosis_code": ["J18.9 - Pneumonia, unspecified organism"]
    },
    # Neurology
    "migraine": {
        "symptoms": ["throbbing headache", "sensitivity to light (photophobia)", "sensitivity to sound (phonophobia)", "nausea", "aura"],
        "history": ["Family history of migraines", "History of aura with headaches", "Triggered by stress or certain foods"],
        "diagnosis_code": ["G43.909 - Migraine, unspecified", "G43.109 - Migraine with aura"]
    },
    "stroke": {
        "symptoms": ["sudden numbness or weakness in the face, arm, or leg, especially on one side", "sudden confusion", "trouble speaking", "sudden severe headache"],
        "history": ["Atrial fibrillation", "Hypertension", "High cholesterol", "Smoker", "Diabetes"],
        "diagnosis_code": ["I63.9 - Cerebral infarction, unspecified (Ischemic Stroke)"]
    },
    # Mental Health
    "depression": {
        "symptoms": ["persistent sad mood", "loss of interest or pleasure", "fatigue", "sleep disturbances", "feelings of worthlessness", "difficulty concentrating"],
        "history": ["Family history of depression", "Major life stressor (e.g., job loss, grief)", "Chronic illness", "Social isolation"],
        "diagnosis_code": ["F32.9 - Major depressive disorder, single episode, unspecified", "F33.1 - Major depressive disorder, recurrent, moderate"]
    },
    "anxiety_disorder": {
        "symptoms": ["excessive worry", "restlessness", "feeling on-edge", "panic attacks", "avoidance of social situations"],
        "history": ["History of panic attacks", "Traumatic event", "Family history of anxiety"],
        "diagnosis_code": ["F41.1 - Generalized anxiety disorder", "F41.0 - Panic disorder"]
    },
    # Gastroenterology
    "gerd": {
        "symptoms": ["heartburn", "acid regurgitation", "chest pain", "chronic cough", "difficulty swallowing"],
        "history": ["Obesity", "Smoker", "Diet high in fatty or spicy foods", "Hiatal hernia"],
        "diagnosis_code": ["K21.9 - Gastro-esophageal reflux disease without esophagitis"]
    },
    "ibs": {
        "symptoms": ["abdominal pain or cramping", "bloating", "gas", "alternating diarrhea and constipation"],
        "history": ["High-stress lifestyle", "Known food sensitivities or intolerances", "Family history of IBS"],
        "diagnosis_code": ["K58.0 - Irritable bowel syndrome with diarrhea", "K58.9 - Irritable bowel syndrome without diarrhea"]
    },
    # Musculoskeletal
    "osteoarthritis": {
        "symptoms": ["joint pain", "stiffness, especially in the morning", "decreased range of motion", "swelling", "grating sensation"],
        "history": ["Older age", "Previous joint injury", "Obesity", "Repetitive stress on joint"],
        "diagnosis_code": ["M19.90 - Unspecified osteoarthritis, unspecified site"]
    },
    # Dermatology
    "atopic_dermatitis": {
        "symptoms": ["dry, scaly skin", "itching", "red to brownish-gray patches", "small, raised bumps which may leak fluid"],
        "history": ["Personal or family history of eczema, allergies, hay fever, or asthma"],
        "diagnosis_code": ["L20.9 - Atopic dermatitis, unspecified"]
    },
    # Urology
    "kidney_stones": {
        "symptoms": ["severe pain in side and back", "pain that radiates to the lower abdomen and groin", "pain on urination", "pink, red or brown urine"],
        "history": ["Chronic dehydration", "Diet high in protein, sodium, or sugar", "Family history of kidney stones"],
        "diagnosis_code": ["N20.0 - Calculus of kidney (Kidney Stone)"]
    },
    # Oncology (Basic)
    "skin_cancer_screening": {
        "symptoms": ["new mole or growth", "a sore that does not heal", "change in an existing mole's size, shape, or color"],
        "history": ["History of excessive sun exposure or tanning bed use", "Fair skin", "Multiple blistering sunburns"],
        "diagnosis_code": ["C44.90 - Other and unspecified malignant neoplasm of skin, unspecified (Basal Cell Carcinoma)"]
    },
    # General / Routine
    "routine_checkup": {
        "symptoms": ["none", "general fatigue", "requests routine blood work"],
        "history": ["No significant past medical history", "Active lifestyle", "Non-smoker"],
        "diagnosis_code": ["Z00.00 - Encounter for general adult medical examination without abnormal findings"]
    }
}

def generate_patient_record(patient_id):
    """Generates a single, random but consistent patient record."""
    
  
    profile_name = random.choice(list(PROFILES.keys()))
    profile = PROFILES[profile_name]
    
 
    symptoms = ", ".join(random.sample(profile["symptoms"], k=random.randint(1, len(profile["symptoms"]))))
    history = random.choice(profile["history"])
    diagnosis = random.choice(profile["diagnosis_code"])
    
    return [
        f"P{patient_id:09d}",  
        symptoms,
        history,
        diagnosis
    ]

def main(num_records):
    """Main function to generate data and write to a CSV file."""
    
    output_file = 'large_patient_dataset.csv'
    headers = ['patient_id', 'symptoms', 'history', 'diagnosis_code']
    
    print(f"Starting data generation for {num_records} records...")
    start_time = time.time()
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(1, num_records + 1):
            writer.writerow(generate_patient_record(i))
            
            if i % 10000 == 0:
                print(f"  ... generated {i}/{num_records} records")

    end_time = time.time()
    print(f"\nSuccessfully generated {num_records} records.")
    print(f"Data saved to '{output_file}'")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")



if __name__ == "__main__":

    NUMBER_OF_RECORDS_TO_GENERATE = 20000 
    main(NUMBER_OF_RECORDS_TO_GENERATE)