import torch
import joblib
from job_posting import JobPosting
from network import Network

VECTOR_FILE = "vectorizers.pkl"
MODEL_FILE = "saved_model.pt"


def load_vectorizers():
    return joblib.load(VECTOR_FILE)


def load_model():
    model = Network()
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
    model.eval()
    return model


def vectorize_job_posting(job_posting, vectorizers):
    # print("Available vectorizer keys:", vectorizers.keys())
    # Apply TF-IDF transformations
    features = []
    features += vectorizers["title"].transform([job_posting.title]).toarray()[0].tolist()
    features += vectorizers["location"].transform([job_posting.location]).toarray()[0].tolist()
    features += vectorizers["department"].transform([job_posting.department]).toarray()[0].tolist()
    features += vectorizers["company_profile"].transform([job_posting.company_profile].toarray()[0].tolist()
    features += vectorizers["description"].transform([job_posting.description].toarray()[0].tolist()
    features += vectorizers["requirement"].transform([job_posting.requirement]).toarray()[0].tolist()
    features += vectorizers["benefit"].transform([job_posting.benefit]).toarray()[0].tolist()
    features.append(int(job_posting.telecommuting))
    features.append(int(job_posting.has_company_logo))
    features.append(int(job_posting.has_question))
    features += vectorizers["employment_type"].transform([job_posting.employment_type]).toarray()[0].tolist()
    features += vectorizers["required_experience"].transform([job_posting.required_experience]).toarray()[0].tolist()
    features += vectorizers["required_education"].transform([job_posting.required_education]).toarray()[0].tolist()
    features += vectorizers["industry"].transform([job_posting.industry]).toarray()[0].tolist()
    features += vectorizers["function"].transform([job_posting.function]).toarray()[0].tolist()

    # Parse salary_range
    if isinstance(job_posting.salary_range, str) and "-" in job_posting.salary_range:
        try:
            low, high = map(int, job_posting.salary_range.split("-"))
            features += [low, high]
        except:
            features += [0, 0]
    else:
        features += [0, 0]
    # try:
    #     features.append(int(job_posting.fraudulent) if job_posting.fraudulent != "" else 0)
    # except ValueError:
    #     features.append(0)  # Default to 0 if fraudulent is not a valid integer
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def predict(job_posting):
    vectorizers = load_vectorizers()
    model = load_model()
    vectorized_input = vectorize_job_posting(job_posting, vectorizers)
    with torch.no_grad():
        output = model(vectorized_input)
        prediction = torch.argmax(torch.softmax(output, dim=1)).item()
        return "Real" if prediction == 0 else "Fake"


if __name__ == "__main__":
    Example test case
    sample_job = JobPosting(
            title="Online Survey Taker – Earn ₹5,000/day!", #fake
            location="Work from Anywhere",
            department="Customer Feedback",
            salary_range="₹5,000 - ₹10,000 daily",
            company_profile="We are a global insights firm seeking individuals to give feedback on daily products. No previous experience required.",
            description="Complete simple online surveys and earn money. No skills required. Just 15 minutes per day needed!",
            requirement="Internet connection, basic reading ability, willingness to complete 5-10 surveys daily.",
            benefit="Daily payments via Paytm or UPI, flexible hours, work from mobile or computer.",
            telecommuting="1",
            has_company_logo="0",
            has_question="0",
            employment_type="Part-time",
            required_experience="No experience",
            required_education="No formal education",
            industry="Market Research",
            function="Data Collection"
        )

    result = predict(sample_job)
    print("Prediction for sample job:",result)


