from sentence_transformers import SentenceTransformer, util

# Load transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------- KNOWLEDGE BASE --------------------

qa_pairs = [

    # -------------------- SYMPTOMS --------------------
    ("I have fever",
     "Drink plenty of water, take rest, and monitor your temperature. If it lasts more than 2 days, consult a doctor."),

    ("My body feels hot",
     "Drink plenty of water, take rest, and monitor your temperature. If it lasts more than 2 days, consult a doctor."),

    ("I have headache",
     "Rest in a quiet place, stay hydrated, and avoid screen time. If pain is severe, seek medical advice."),

    ("I have cold",
     "Drink warm fluids, take rest, and try steam inhalation. If symptoms worsen, consult a doctor."),

    ("I have cough",
     "Stay hydrated, take warm drinks, and avoid cold foods. If it continues for more than a week, consult a doctor."),

    ("I feel tired",
     "Ensure proper sleep, eat healthy food, and stay hydrated. Persistent tiredness may need medical attention."),

    # -------------------- MEDICINES --------------------
    ("What is paracetamol",
     "Paracetamol is used to reduce fever and relieve mild to moderate pain."),

    ("What is ibuprofen",
     "Ibuprofen is used to reduce pain, inflammation, and fever. It is better taken with food to avoid stomach irritation."),

    ("What is cetirizine",
     "Cetirizine is used to relieve allergy symptoms like sneezing, runny nose, and itching."),

    ("What is ORS",
     "ORS (Oral Rehydration Solution) helps prevent dehydration, especially during diarrhea or vomiting."),

    ("What is antacid",
     "Antacids help relieve acidity, heartburn, and indigestion."),

    ("medicine for pain",
     "Mild pain relievers like paracetamol or ibuprofen can help. Avoid overuse and follow proper guidance."),

    ("medicine for allergy",
     "Antihistamines like cetirizine can help relieve allergy symptoms."),

    ("acidity treatment",
     "Antacids can help relieve acidity and heartburn."),

    ("how to treat dehydration",
     "Drink plenty of fluids. ORS is commonly used to restore lost fluids and electrolytes."),

    # -------------------- DIET --------------------
    ("What is healthy diet",
     "Include fruits, vegetables, whole grains, and protein-rich foods in your diet."),

    ("How much water should I drink",
     "Drink around 6–8 glasses of water daily depending on your activity level."),

    # -------------------- LIFESTYLE --------------------
    ("How much sleep needed",
     "Adults typically need 7–9 hours of sleep per night."),

    ("Why exercise is important",
     "Exercise improves heart health, maintains weight, and boosts mood."),

    # -------------------- MENTAL HEALTH --------------------
    ("How to reduce stress",
     "Practice deep breathing, meditation, and take breaks."),

    ("Why am I anxious",
     "Anxiety can be caused by stress or lack of sleep. Try relaxation techniques or talk to someone."),

    # -------------------- FIRST AID --------------------
    ("First aid for cuts",
     "Clean the wound with water, apply antiseptic, and cover with a bandage."),

    ("Burn treatment",
     "Cool the burn under running water and avoid applying ice directly."),
]

questions = [q for q, a in qa_pairs]
answers = [a for q, a in qa_pairs]

# Precompute embeddings
question_embeddings = model.encode(questions, convert_to_tensor=True)

# -------------------- RULE-BASED SAFETY --------------------

def rule_based_response(user_input):
    text = user_input.lower()
    if any(word in text for word in ["hi", "hello", "hey"]):
        return "Hello! How can I assist you with your health today?"
    # Severity detection
    if any(word in text for word in ["severe", "not stopping", "3 days", "worsening"]):
        return "This may need medical attention. Please consult a doctor soon."

    # Medicine misuse protection
    if any(word in text for word in ["overdose", "too much medicine"]):
        return "Avoid taking too much medicine. Please consult a doctor immediately."

    return None

# -------------------- MAIN FUNCTION --------------------

def get_response(user_input):

    # 1️. Safety first
    rule = rule_based_response(user_input)
    if rule:
        return rule

    # 2️. Convert user input to embedding
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # 3️. Compute similarity
    scores = util.cos_sim(user_embedding, question_embeddings)

    best_idx = scores.argmax()
    best_score = scores[0][best_idx]

    # 4️.Threshold check
    if best_score < 0.5:
        return "I’m not sure I understood. Please describe your symptom clearly."

    return answers[best_idx]