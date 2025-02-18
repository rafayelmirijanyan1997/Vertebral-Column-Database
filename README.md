# Vertebral Column Dataset Analysis

## 📌 Overview
This repository contains a Jupyter Notebook (**Vertebral_Column_dataset_analysis.ipynb**) that performs an analysis on the **Vertebral Column dataset**. The dataset is fetched from the **UCI Machine Learning Repository** and includes biomechanical features used to classify orthopedic patients into different categories (**Normal, Disk Hernia, or Spondylolisthesis**). 

The analysis includes:
- **Data Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Application of Machine Learning techniques** (K-Nearest Neighbors - KNN) for classification.

---

## 📂 Dataset
The dataset used in this analysis is the **Vertebral Column dataset** (ID: 212) from the **UCI Machine Learning Repository**. It contains six **biomechanical features** derived from the shape and orientation of the **pelvis and lumbar spine**.

### **Features:**
1. **Pelvic Incidence**
2. **Pelvic Tilt**
3. **Lumbar Lordosis Angle**
4. **Sacral Slope**
5. **Pelvic Radius**
6. **Degree of Spondylolisthesis**

### **Target Variable:**
- **Class:** Normal, Hernia, or Spondylolisthesis

---

## 🔧 Requirements
To run the Jupyter Notebook, install the following **Python libraries**:

- `ucimlrepo` → To fetch the dataset from UCI Machine Learning Repository.
- `numpy` → For numerical operations.
- `pandas` → For data manipulation and analysis.
- `matplotlib` → For data visualization.
- `seaborn` → For advanced data visualization.
- `scikit-learn` → For machine learning algorithms and evaluation metrics.

### **Installation:**
```sh
pip install ucimlrepo numpy pandas matplotlib seaborn scikit-learn
```

---

## 📑 Notebook Structure
### **1️⃣ Data Downloading and Loading**
- Fetch dataset using **ucimlrepo**.
- Display basic dataset information, including **feature descriptions and metadata**.

### **2️⃣ Data Preprocessing & Exploratory Data Analysis (EDA)**
- Handle **missing values** and **encode categorical variables**.
- Visualize relationships between **features** and **target classes**.

### **3️⃣ Machine Learning Model Application**
- Apply **K-Nearest Neighbors (KNN) classifier**.
- Evaluate model using:
  - **Accuracy**
  - **F1 Score**
  - **Confusion Matrix**

### **4️⃣ Results and Visualizations**
- Visualize results with **confusion matrices** and **other relevant plots**.
- Discuss insights derived from the classification results.

---

## 🚀 Usage
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/rafayelmirijanyan1997/Vertebral-Column-Database
cd vertebral-column-database
```

### **2️⃣ Install Dependencies**
Ensure you have the required Python libraries installed as mentioned in the **Requirements** section.

### **3️⃣ Run the Jupyter Notebook**
Open and run the Jupyter Notebook sequentially:
```sh
jupyter notebook Vertebral_Column_dataset_analysis.ipynb
```

---

## 📥 Download the Notebook
You can download the Jupyter Notebook directly from this repository by clicking on the **Vertebral_Column_dataset_analysis.ipynb** file and selecting the **"Download"** option.

---

## 📜 License
This project is licensed under the **MIT License**. See the **LICENSE** file for details.

---

## 🙌 Acknowledgments
- The dataset is provided by the **UCI Machine Learning Repository**.
- The analysis is inspired by various **machine learning and data science resources**.

---

## 📬 Contact
For any **questions** or **suggestions**, please feel free to **open an issue** or contact the repository owner.

📌 **Note:** Ensure you have the necessary permissions to use and distribute the dataset as per the **UCI Machine Learning Repository’s** terms of use.
