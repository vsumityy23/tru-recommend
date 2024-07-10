
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import ast

def get_embeddings(text,tokenizer,model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def recommend_students(project_embedding,resumes_df,embedding_name, top_n=10):
    
    project_embedding = project_embedding.reshape(1, -1)  # Reshape to ensure it's 2D
    student_embeddings =torch.stack([torch.tensor(embed) for embed in resumes_df[embedding_name].tolist()])
    # Convert student_embeddings to a 2D numpy array
    student_embeddings_np = student_embeddings.numpy().reshape(len(student_embeddings), -1)
    similarities = cosine_similarity(project_embedding, student_embeddings_np)
    similarities = similarities.flatten()  # Flatten to 1D array
    top_students_idx = similarities.argsort()[-top_n:][::-1]
    return resumes_df.iloc[top_students_idx]


def recommend_projects(student_embedding,projects_df, embedding_name,top_n=10):

     # Reshape the embeddings to 2D arrays for cosine_similarity
    student_embedding = student_embedding.reshape(1, -1)  # Reshape to ensure it's 2D
    project_embeddings = torch.stack([torch.tensor(embed) for embed in projects_df[embedding_name].tolist()])
    project_embeddings = project_embeddings.numpy().reshape(len(project_embeddings), -1)
    similarities = cosine_similarity(student_embedding, project_embeddings)[0]
    top_projects_idx = similarities.argsort()[-top_n:][::-1]
    return projects_df.iloc[top_projects_idx]

from transformers import BertTokenizer, BertModel,GPT2Tokenizer,GPT2Model,RobertaTokenizer,RobertaModel

# Load pre-trained model and tokenizer

#bert 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

#gpt2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  #since gpt2 doesn't have default padding so we are padding the tokens 
gpt2_model = GPT2Model.from_pretrained('gpt2')

#roberta
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')



#list of skills name
skills = [
        "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "PHP", "Swift", 
        "Kotlin", "Go", "R", "MATLAB", "Perl",
        "HTML", "CSS", "React.js", "Angular.js", "Vue.js", "Node.js", "Django", 
        "Flask", "Ruby on Rails", "ASP.NET",
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQLite", "Firebase",
        "Git", "GitHub", "GitLab", "Bitbucket",
        "Object-Oriented Programming (OOP)", "Functional Programming", 
        "Agile Methodologies", "Test-Driven Development (TDD)", 
        "Continuous Integration/Continuous Deployment (CI/CD)",
        "iOS Development", "Android Development", "React Native", "Flutter",
        "Data Wrangling", "Data Visualization", "Machine Learning", 
        "Deep Learning", "Natural Language Processing (NLP)", "Big Data",
        "AWS", "Azure", "Google Cloud Platform (GCP)", "Docker", "Kubernetes", 
        "Jenkins",
        "Network Security", "Application Security", "Ethical Hacking", 
        "Penetration Testing"
      ]

#list of tools name
tools = [
        "Visual Studio Code", "PyCharm", "IntelliJ IDEA", "Sublime Text", "Atom", 
        "Eclipse",
        "Git", "GitHub", "GitLab", "Bitbucket",
        "Jira", "Trello", "Asana", "Monday.com",
        "Jenkins", "Travis CI", "CircleCI", "GitLab CI/CD",
        "Docker", "Kubernetes", "OpenShift",
        "Amazon Web Services (AWS)", "Microsoft Azure", 
        "Google Cloud Platform (GCP)", "IBM Cloud",
        "MySQL Workbench", "pgAdmin", "MongoDB Compass", 
        "SQLite Database Browser",
        "Jupyter Notebook", "Anaconda", "TensorFlow", "Keras", "PyTorch", 
        "Scikit-learn",
        "Prometheus", "Grafana", "ELK Stack (Elasticsearch, Logstash, Kibana)", 
        "Splunk",
        "Postman", "Swagger", "SoapUI"
      ]

    # List of Certifications
certifications = [
        "Microsoft Certified: Azure Fundamentals",
        "Oracle Certified Professional: Java SE Programmer",
        "Python Institute: PCEP, PCAP",
        "FreeCodeCamp Certifications",
        "W3Schools Certifications",
        "Udemy Web Development Bootcamps",
        "IBM Data Science Professional Certificate (Coursera)",
        "Google Professional Data Engineer",
        "Microsoft Certified: Azure Data Scientist Associate",
        "AWS Certified Solutions Architect",
        "Google Certified Professional Cloud Architect",
        "Microsoft Certified: Azure Solutions Architect Expert",
        "Certified Ethical Hacker (CEH)",
        "CompTIA Security+",
        "Certified Information Systems Security Professional (CISSP)",
        "AWS Certified DevOps Engineer",
        "Docker Certified Associate",
        "Certified Kubernetes Administrator (CKA)",
        "Google Associate Android Developer",
        "Apple Certified iOS Developer"
      ]
    
    # List of Project Names
projects = [
    "Personal Portfolio Website",
    "E-commerce Platform",
    "Chat Application",
    "Social Media Dashboard",
    "Task Management System",
    "Blog Website",
    "Online Learning Platform",
    "Weather Forecast Application",
    "Fitness Tracker App",
    "Budget Tracker",
    "Recipe Sharing Platform",
    "Job Board Website",
    "Event Management System",
    "Travel Booking Website",
    "Cryptocurrency Dashboard",
    "Online Forum",
    "Content Management System",
    "Real-Time Collaboration Tool",
    "Inventory Management System",
    "Library Management System",
    "Quiz Application",
    "News Aggregator",
    "Online Marketplace",
    "Survey Tool",
    "Customer Relationship Management (CRM) System",
    "Project Management Tool",
    "Restaurant Reservation System",
    "Online Banking System",
    "Healthcare Management System",
    "Employee Management System",
    "Hotel Booking System",
    "Virtual Classroom",
    "Online Exam Portal",
    "Stock Market Analysis Tool",
    "Music Streaming Service",
    "Video Sharing Platform",
    "Food Delivery App",
    "Taxi Booking System",
    "IoT Home Automation",
    "Blockchain-Based Voting System"
    ]