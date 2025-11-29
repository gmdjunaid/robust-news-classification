### Instructions for editing (for humans and AI)

- **Purpose**: This file tracks development-session notes and high-level decisions for the robust news classification project.
- **Heading style**:
  - Use `###` for main sections and `####` for individual dev sessions inside the `Dev session log`.
  - Do not use `#` headings.
- **Bullet style**:
  - Prefer bullets with bold \"pseudo-headings\", e.g. `- **item**: detail`.
  - Keep language concise and neutral.
- **Sections to keep**:
  - `Project summary`
  - `Data organization so far`
  - `Challenges`
  - `Current plan`
  - `Future notes / TODOs`
  - `Dev session log`
- **How to add new notes (humans and AI)**:
  - For new work in a specific session, add a new dated entry under `Dev session log` and describe changes as short bullets.
  - Append new dev sessions to the **bottom** of the log without rewriting or deleting earlier sessions.
  - Only update the high-level sections above if something has materially changed (e.g., project goal, data layout, main plan).
  - When structure needs to evolve, preserve existing sections and add new ones in the same style instead of renaming or removing old ones.

### Dev session log

#### 2025-11-29 – Data layout & large CSV planning

- **Training vs. test data**: Decided how to separate and store CSVs for training (`training-data`) versus testing (`test-data`).
- **File locations**:
  - Training: `training-data/Fake.csv`, `training-data/True.csv`.
  - Test: `test-data/fake.csv`, plus a very large external CSV for additional test examples.
- **Git performance issue**: Noted that large CSV sizes make git push/pull operations slow and cumbersome.
- **Large external CSV**: Identified that the extra test CSV is too large to use fully for quick experiments.
- **Plan for sampling**: Agreed that the first step is to clean and downsample the large external test CSV into a smaller, representative sample to quickly evaluate fake vs. real classification accuracy.

#### 2025-11-29 – `.gitignore` for large test CSV

- **Ignore huge CSV**: Added `test-data/WELFake_Dataset.csv` to `.gitignore` so the very large test CSV is not tracked by git, improving push/pull performance.
- **Temporary workaround**: Treat this ignore rule as a temporary solution while we experiment; we may later replace the raw file with a smaller derived version that can be tracked.
- **Future direction**: Still planning to create a smaller, downsampled version of the large test CSV, since we do not need the full dataset for current accuracy experiments.

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
  - Decision on whether to keep ignoring the original huge CSV file or replace it with a smaller, tracked subset file.
  - **TODO (partner)**: Write a `scripts/prepare_welfake_sample.py` (or similar) script that:
    - Assumes each developer has the original large WELFake CSV locally.
    - Cleans/reduces that CSV by taking (for now) the first 1,000 rows.
    - Saves the result as a smaller CSV in `test-data/` to be used as test data.
    - Keeps the script in the repo so reviewers can see exactly how the large CSV was processed.

