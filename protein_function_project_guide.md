## Protein Function Prediction — Simple Guide for Non‑Experts

This is a **plain‑language guide** to your protein function prediction project.  
Imagine you are explaining it to someone who:

- Has **no programming background**
- Has **no biology background**
- Just wants to understand **what is happening and why**

We will:

- Explain **every big idea** in simple terms.
- Walk through the **whole pipeline** from raw data to predictions.
- Include **one clear case study** at the end to make it real.

---

## 1. What Are Proteins?

- **Proteins** are tiny **machines** inside all living things.
- They help with almost **everything** in the body:
  - Moving oxygen in blood  
  - Breaking down food  
  - Sending signals (like hormones)  
  - Building and repairing tissues  

### 1.1 Proteins as “letter strings”

- A protein is made from **amino acids** (20 types).
- Each amino acid is written as **one letter**, like:
  - `A`, `C`, `D`, `E`, `F`, `G`, … (total 20 different letters)
- So a protein looks like a **long string of letters**.  
  Example:

  `MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESF...`

- You can think of this like:
  - A **sentence** made of special letters.
  - The **order** of letters decides:
    - The **shape** of the protein
    - Its **job** (function)

Examples of protein functions:

- **Hemoglobin**: carries oxygen in the blood.
- **Enzymes**: help break down food.
- **Insulin**: helps control blood sugar.

In many cases, scientists know the **letters** of a protein, but they do **not yet know its job**.  
Your project tries to **guess the job** of a protein from its letters using AI.

---

## 2. What Is This Project Trying To Do?

### 2.1 The main question

We want the computer to answer:

> “If I give you the **letters of a protein**, can you tell me what **job** it does?”

Scientists already know the function of **some** proteins, because:

- These proteins have been studied in the lab.
- Their functions are written in big **databases**.

But for **many** new proteins, the function is **unknown**.  
It is too slow and expensive to test all of them in the lab.

So we use **AI** (machine learning) to:

- Learn from **known** proteins
- Then **predict** the functions of **new** proteins

---

## 3. How Do We Describe a Protein’s Function?

### 3.1 GO terms (Gene Ontology)

Scientists use codes called **GO terms** (Gene Ontology IDs) to describe functions.

Examples:

- `GO:0005829` – cytosol (inside of the cell fluid)
- `GO:0005737` – cytoplasm
- `GO:0005886` – plasma membrane
- `GO:0005634` – nucleus

One protein can have **several** GO terms.  
This is like saying:

- One person can be:
  - A **teacher**
  - A **parent**
  - A **musician**

So our prediction problem is:

- Not just “pick one label”.
- It is **multi‑label**: for each GO term, decide **yes or no** for that protein.

---

## 4. Big Picture of the Pipeline

Your project is organized into **steps**, each step in its own notebook:

- `01_preprocess.ipynb` – prepare and clean the data
- `02_feature_extraction.ipynb` – turn sequences into numeric features
- `03_train.ipynb` – train the prediction model
- `04_evaluate_and_results.ipynb` – measure how good the model is
- `05_predict.ipynb` – use the model on new proteins
- `06_visualize.ipynb` – advanced visualizations and plots

Think of this like a **factory**:

1. **Raw material** (downloaded data)
2. **Cleaning** (filtering and organizing)
3. **Feature making** (turn letters into numbers)
4. **Training** (teaching the model)
5. **Testing** (checking how good it is)
6. **Using it** (making predictions)
7. **Visualizing** (seeing patterns)

---

## 5. Step 1 – Collecting and Preparing Data

### 5.1 Where does the data come from?

- The script `collect_data.py` talks to **UniProt**, a large public protein database.
- It asks:
  - “Give me proteins that:
    - Are **high quality / reviewed**
    - Have **known GO terms**
    - Have sequence lengths in a reasonable range”

### 5.2 What files are created?

The script creates several CSV files (tables):

- **`proteins.csv`**
  - Each row = **one protein**
  - Columns: `protein_id`, `protein_name`, `organism`, `sequence`, `length`
- **`go_annotations.csv`**
  - Each row = one link between a **protein** and a **GO term**
  - Shows which protein has which GO terms
- **`protein_go_summary.csv`**
  - For each protein:
    - How many GO terms it has
    - Which GO terms they are
- **`go_term_statistics.csv`**
  - For each GO term:
    - How often it appears
    - How many proteins have it
- **`sequence_statistics.csv`**
  - Basic info about sequences:
    - Length
    - Which amino acids are common, etc.

This is like:

- Building a clean **spreadsheet** that says:
  - “This protein has these functions (GO terms).”

---

## 6. Step 2 – Turning Sequences into Numbers (Features)

Computers need **numbers**, not letters.

### 6.1 What is a “protein language model”?

You may have heard about:

- ChatGPT, Google Translate, etc.

These use **language models** trained on many sentences.

A **protein language model** does the same, but:

- It is trained on **millions of protein sequences** instead of English.

In your project, you use **ESM‑2** (a protein language model from Meta/Facebook).

### 6.2 What is an “embedding”?

When you send a protein sequence into ESM‑2:

1. It reads the sequence, like reading a sentence.
2. It outputs a long list of **numbers** (for example, 320 numbers).
3. This list is called an **embedding**:
   - It represents the protein in a way the model understands.
   - Proteins with **similar functions** tend to have **similar embeddings**.

You save these embeddings into files:

- `embeddings_train.npy`
- `embeddings_val.npy`
- `embeddings_test.npy`

You also save **labels** as arrays:

- `labels_train.npy`
- `labels_val.npy`
- `labels_test.npy`

---

## 7. Step 3 – Training the Classifier (03_train.ipynb)

Now that each protein is a list of numbers (embedding), we train a **smaller network** to map:

> embedding → GO functions

### 7.1 The small neural network

This model:

- Takes input: **320‑dimensional embedding** (320 numbers)
- Passes through **hidden layers**:
  - These layers help learn complex patterns.
  - They use **residual blocks**, which help the model train more smoothly.
- Outputs: for each GO term (e.g., 4 GO terms), it gives a **score**.
  - After a **sigmoid** function, each score becomes a **probability between 0 and 1**.

Interpretation:

- If the output for a GO term is **0.80**, that means:
  - “The model thinks there is an 80% chance that this protein has this GO term.”

Because a protein can have **several** GO terms, this is **multi‑label**:

- For each GO term, we say **yes/no** separately, instead of choosing just one.

### 7.2 Loss function: Focal loss (why we use it)

We want the model to learn from mistakes.  
But some labels (GO terms) might be **rare**.

- If we used a plain loss:
  - The model might **ignore rare labels**.
- Focal loss:
  - Puts **extra focus** on examples the model is getting wrong.
  - Helps the model pay attention to **hard and rare cases**.

In simple words:

- If the model keeps messing up on certain labels,
- Focal loss tells it: “These difficult ones are important, learn them better.”

### 7.3 Training loop

Training happens in **epochs** (rounds):

For each epoch:

1. Go through all training proteins in **small batches**.
2. For each batch:
   - Get embeddings and true labels.
   - Run through the model.
   - Compute the **loss**.
   - Adjust model weights (learning).
3. After one full pass:
   - Check performance on the **validation set**:
     - Compute:
       - **Validation loss**
       - **Validation F1** (micro and macro)
       - **Validation AUROC**
       - **Validation accuracy** (you have added this)

We also use:

- **Early stopping**:
  - If validation F1 stops getting better for a certain number of epochs,
  - We **stop training early** to avoid overfitting.
- **Best model saving**:
  - Whenever the validation micro‑F1 is the best so far, we:
    - Save the model weights to `best_model.pt`
    - Save the associated **best validation accuracy** as well.

At the end, the notebook prints something like:

- `TRAINING COMPLETE — Best F1µ: X.XXXX | Best Val Accuracy: Y.YYYY`

---

## 8. Step 4 – Evaluation (04_evaluate_and_results.ipynb)

Now we want to know:

> “How good is the model really?”

To test this, we use the **test set**:

- Data the model has **never** seen during training.

### 8.1 What do we compute?

We load:

- The best model (`best_model.pt`)
- The test embeddings and labels

We get **probabilities** for each GO term, then turn them into 0/1 predictions using a threshold (usually 0.5).

Then we compute metrics:

- **F1 (micro)**:
  - Good overall measure, combining precision and recall across all labels.
- **F1 (macro)**:
  - Treats each GO term equally, good to see if some rare labels are bad.
- **Precision and recall**:
  - Precision: “Of the labels we predicted positive, how many were actually positive?”
  - Recall: “Of all actual positive labels, how many did we find?”
- **Hamming loss**:
  - How many label decisions were wrong on average.
  - Lower is better.
- **Accuracy** (you now added it):
  - Defined as `1 − Hamming loss`.
  - You can read it as:
    - “Out of all individual label decisions, what fraction did we get right?”
- **Subset accuracy**:
  - Very strict:
    - A sample is counted correct **only if all its labels are exactly correct**.
- **AUROC (micro and macro)**:
  - Measures how well the model can rank positives vs negatives.
- **Mean Average Precision (MAP)**:
  - Looks at quality of ranking for each label and averages.

The notebook:

- Saves these metrics to:
  - `results/metrics/evaluation_report.json`
  - `results/metrics/summary_table.csv`
- Also prints them in a **nice table** for you to read.

---

## 9. Step 5 – Making Predictions on New Proteins (05_predict.ipynb)

Once the model is trained and evaluated, we can **use it on new sequences**.

### 9.1 How it works

For a new protein sequence:

1. Use the same ESM‑2 model to get an **embedding**.
2. Feed that embedding into the trained **classifier**.
3. Get probabilities for each GO term.
4. Decide:
   - Which GO terms are “high confidence” (above some threshold).
5. Save:
   - Predictions to `results/metrics/predictions.json`
   - A plot showing **confidence** of each predicted GO term.

This notebook also has:

- Example proteins (like Hemoglobin, GFP, Insulin).
- Shows confidence bars as a **bar chart**.
- Lets you paste in **your own sequence** to test.

---

## 10. Step 6 – Visualizations (06_visualize.ipynb)

This notebook helps you **see** what is going on.

Some typical visualizations:

- **Embedding plots** (e.g., using UMAP):
  - Reduce the high‑dimensional embeddings (320 numbers) down to 2D or 3D.
  - Plot them as dots.
  - Color by GO term or other properties.
- **Per‑class F1 plots**:
  - Which GO terms are easy/hard for the model?
- **ROC and PR curves**:
  - Show model performance across different thresholds.
- **Confidence distributions**:
  - See if the model is over‑confident or under‑confident.

These pictures are very helpful when explaining the model to others (for presentations, reports, etc.).

---

## 11. Case Study: Hemoglobin β (Real Example)

Let’s walk through **one protein** as a full story:

### 11.1 Meet the protein

- Name: **Hemoglobin β (beta chain of hemoglobin)**
- Job: Carries oxygen in red blood cells.

In the notebook, you may have a sequence like:

- `'Hemoglobin β': 'MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESF...'`  
  (shortened here for readability)

This is the **letter sequence** of the protein.

### 11.2 What happens in the project?

1. **Data collection phase**  
   - UniProt gives us Hemoglobin β with:
     - Its full sequence
     - GO terms describing its function and location.

2. **Preprocessing**  
   - It is included in the dataset.
   - It has one or more GO terms, such as:
     - `GO:0005829` (cytosol)
     - `GO:0005737` (cytoplasm)
     - etc.

3. **Feature extraction**  
   - The sequence is sent into **ESM‑2**.
   - ESM‑2 outputs an **embedding** (320 numbers).
   - This embedding is saved in `embeddings_train.npy` (or val/test, depending on the split).

4. **Training**  
   - The classifier sees the embedding and the true GO labels.
   - Over many epochs, it learns:
     - “Embeddings like this usually correspond to these GO terms.”

5. **Evaluation**  
   - If Hemoglobin β is in the test set:
     - The model does **not** see it during training.
     - Later, when evaluating, the model predicts GO terms for it.
     - We compare:
       - **Predicted labels** vs **true labels**.
     - This contributes to the **F1 score and accuracy**.

6. **Prediction notebook (05_predict.ipynb)**  
   - When you run the example section:
     - The notebook prints something like:
       - For Hemoglobin β (147 aa) — several predictions above a threshold:
         - GO:0005829  (with some confidence, e.g. 0.56)
         - GO:0005737
         - GO:0005886
         - GO:0005634
   - It also makes a **bar chart**:
     - Each bar = one GO term
     - Bar length = confidence

7. **Visualization**  
   - In `06_visualize.ipynb`, you could see Hemoglobin β as a dot on a 2D plot:
     - Near other proteins with similar functions.
     - Colored by one of its GO terms.

### 11.3 What does this tell us?

- Without doing a **wet‑lab experiment**, the model:
  - Sees only the **sequence** of Hemoglobin β.
  - Uses patterns learned from **many other proteins**.
  - Predicts functions that match the real biological understanding.
- This shows:
  - How **sequence alone**, plus a **pretrained language model**, can be powerful.
  - How this pipeline can help **prioritize** proteins for further experimental study.

