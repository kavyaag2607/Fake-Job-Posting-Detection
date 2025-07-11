import torch
import pandas as pd
import random
from job_posting import JobPosting, JobPostingsDataset
import network
import joblib
from pycm import ConfusionMatrix
from skorch import NeuralNetClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

CSV_FILENAME = "job_dataset.csv"
CSV_FILE_DELIMITER = ","
EPOCHS_COUNT = 200
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
KFOLD_PARTITIONS_COUNT = 5
FAKE_TO_REAL_RATIO = 1.0/20.0
TRAINING_PARALLEL_JOBS_COUNT = -1  # -1 means maximum


def prepare_dataset():
    global job_postings
    global all_job_posting_data
    global all_job_posting_targets
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.shuffle(job_postings)
    job_postings = JobPostingsDataset(job_postings)
    job_postings.prepare_all_text_vectorizers()
    all_job_posting_data = torch.tensor([job_posting.get_data_list() for job_posting in job_postings]).float().to(device)
    all_job_posting_targets = torch.tensor([job_posting.get_target_list() for job_posting in job_postings]).float().to(device)


def read_job_postings_data_from_csv(csv_filename=CSV_FILENAME, csv_file_delimiter=CSV_FILE_DELIMITER):
    global job_postings
    job_postings = []
    csv_contents = pd.read_csv(csv_filename, delimiter=csv_file_delimiter, dtype=str)
    print("Size of the dataset is :")
    print(csv_contents.shape)
    # Reads a CSV file into a pandas DataFrame, treating all values as string
    rows = len(csv_contents)
    cols = len(csv_contents.iloc[0])
    for i in range(rows):
        data = csv_contents.iloc[i]
        job_posting = JobPosting(data["title"], data["location"], data["department"], data["salary_range"],
                                 data["company_profile"], data["description"], data["requirements"],
                                 data["benefits"], data["telecommuting"], data["has_company_logo"],
                                 data["has_questions"], data["employment_type"], data["required_experience"],
                                 data["required_education"], data["industry"], data["function"], data["fraudulent"])
        job_postings.append(job_posting)


def initialize_network():
    global model
    global optimizer
    global criterion
    model = network.Network().to(device)
    criterion = torch.nn.BCEWithLogitsLoss
    optimizer = torch.optim.Adam
    for m in model._modules:
        network.normal_init(model._modules[m], 0, 1)


def print_confusion_matrix(actual_labels, predicted_labels):
    cm = ConfusionMatrix(actual_labels, predicted_labels)
    print(cm)


if __name__ == "__main__":
    print("Reading CSV file")
    read_job_postings_data_from_csv()
    print("Preparing dataset")
    prepare_dataset()
    print("Initializing network")
    initialize_network()
    print("Creating classifier")
    classifier = NeuralNetClassifier(network.Network, max_epochs=EPOCHS_COUNT, lr=LEARNING_RATE,
                                     train_split=None, criterion=criterion, optimizer=optimizer,
                                     batch_size=BATCH_SIZE,
                                     criterion__pos_weight=torch.tensor([FAKE_TO_REAL_RATIO, 1.0]))
    print("Training")
    predictions = cross_val_predict(classifier, all_job_posting_data, all_job_posting_targets,
                                    cv=KFOLD_PARTITIONS_COUNT, verbose=1, n_jobs=TRAINING_PARALLEL_JOBS_COUNT)
    predicted_labels = torch.from_numpy(predictions)
    predicted_labels = torch.argmax(predicted_labels, dim=1).tolist()
    actual_labels = torch.argmax(all_job_posting_targets, dim=1).tolist()
    print_confusion_matrix(actual_labels, predicted_labels)
    print("Training input feature size:", all_job_posting_data.shape[1])

    print("Training final model on full dataset...")
    final_model = NeuralNetClassifier(
        network.Network,
        max_epochs=EPOCHS_COUNT,
        lr=LEARNING_RATE,
        train_split=None,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        criterion__pos_weight=torch.tensor([FAKE_TO_REAL_RATIO, 1.0])
    )

    final_model.fit(all_job_posting_data, all_job_posting_targets)

    # Predict on the training data
    train_predictions = final_model.predict(all_job_posting_data)

    # Get actual labels
    actual_labels = torch.argmax(all_job_posting_targets, dim=1).cpu().numpy()

    if len(train_predictions.shape) > 1 and train_predictions.shape[1] > 1:
        train_predictions = np.argmax(train_predictions, axis=1)

    train_accuracy = accuracy_score(actual_labels, train_predictions)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # Save model
    final_model.save_params(f_params="saved_model.pt")
    print("Model saved to saved_model.pt")

    vectorizers = {
        'title': job_postings.title_vectorizer,
        'location': job_postings.location_vectorizer,
        'department': job_postings.department_vectorizer,
        'company_profile': job_postings.company_profile_vectorizer,
        'description': job_postings.description_vectorizer,
        'requirement': job_postings.requirement_vectorizer,
        'benefit': job_postings.benefit_vectorizer,
        'employment_type': job_postings.employment_type_vectorizer,
        'required_experience': job_postings.required_experience_vectorizer,
        'required_education': job_postings.required_education_vectorizer,
        'industry': job_postings.industry_vectorizer,
        'function': job_postings.function_vectorizer
    }

    joblib.dump(vectorizers, 'vectorizers.pkl')
    print("Saved vectorizers to vectorizers.pkl")


