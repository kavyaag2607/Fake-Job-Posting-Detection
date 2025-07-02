from flask import Flask, render_template, request
from job_posting import JobPosting
from predict import predict  # Your prediction logic

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    form_data = {}
    result = None

    if request.method == "POST":
        form = request.form
        form_data = form.to_dict()

        job = JobPosting(
            title=form.get("title", ""),
            location=form.get("location", ""),
            department=form.get("department", ""),
            salary_range=form.get("salary_range", ""),
            company_profile=form.get("company_profile", ""),
            description=form.get("description", ""),
            requirement=form.get("requirements", ""),
            benefit=form.get("benefits", ""),
            telecommuting=form.get("telecommuting", "0"),
            has_company_logo=form.get("has_company_logo", "0"),
            has_question=form.get("has_questions", "0"),
            employment_type=form.get("employment_type", ""),
            required_experience=form.get("required_experience", ""),
            required_education=form.get("required_education", ""),
            industry=form.get("industry", ""),
            function=form.get("function", "")
        )

        result = predict(job)

    return render_template("index.html", result=result, form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True)
