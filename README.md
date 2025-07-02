<h1>Fake Job Posting Detection Using Neural Networks</h1>

<p>
This project helps detect whether a job posting is fake or real using a neural network and deep learning techniques. It assists users in identifying fraudulent job listings with high accuracy.
</p>

<h2>Dataset</h2>

<ul>
  <li><strong>Source:</strong> Kaggle</li>
  <li>
    <strong>Link:</strong>
    <a href="https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction" target="_blank">
      https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
    </a>
  </li>
  <li>The dataset includes fields like job title, company profile, job description, requirements, and more.</li>
  <li>The text data is cleaned and converted into numerical features using TF-IDF vectorization.</li>
</ul>

<h2>Model and Training</h2>

<ul>
  <li>The model is built using PyTorch.</li>
  <li>A simple feedforward neural network is used for binary classification.</li>
  <li>Evaluation is done using a confusion matrix.</li>
  <li>The model achieves an accuracy of approximately 96%.</li>
</ul>

<h2>Web Application</h2>

<p>
A web application is created using Flask to make the model easy to use. It allows users to:
</p>

<ul>
  <li>Enter job posting details manually.</li>
  <li>Submit the details and get a prediction: Fake or Real.</li>
</ul>

<p>This provides a user-friendly way to use the model without running scripts.</p>

<p align="center">
  <img src="https://github.com/kavyaag2607/Fake-Job-Posting-Detection/blob/main/Webpage.png?raw=true" alt="Web Page Preview">
</p>

<h2>How to Run</h2>

<ol>
  <li>Clone the repository:
    <pre><code>git clone https://github.com/kavyaag2607/Fake-Job-Posting-Detection.git
cd Fake-Job-Posting-Detection</code></pre>
  </li>

  <li>Install the required libraries:
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>

  <li>Run the Flask web application:
    <pre><code>python app.py</code></pre>
  </li>

</ol>

