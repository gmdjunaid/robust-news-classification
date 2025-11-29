### Project summary

- **Goal**: Build a model to classify news articles as fake or real.

### Data organization so far

- **Training data location**:
  - `training-data/Fake.csv`
  - `training-data/True.csv`
- **Test data location**:
  - `test-data/fake.csv`
  - An additional, very large CSV file we found for extra test data (not yet fully integrated).
- **Work in progress**: We’ve mainly been deciding where to store training vs. test CSV files and adjusting the project layout to keep this organized.

### Challenges

- **Large CSV sizes** are making git operations (pushing and pulling) slow and a bit cumbersome.
- The **extra test CSV file** is extremely large, which makes it impractical to use in full for quick experiments.

### Current plan

- **Step 1**: Clean and downsample the large external test CSV so that:
  - We keep a smaller, representative sample.
  - We can quickly test how well the model distinguishes fake vs. real news.
- We do **not** need the full dataset yet; this is mainly for an initial accuracy and workflow test.

### Future notes / TODOs

- Use this section to capture next decisions and experiments, for example:
  - Possible sampling strategies for the large test CSV.
  - Notes on model performance and accuracy.
  - Ideas for improving data organization or preprocessing.


