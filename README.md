# Personality-Based Literature Analysis

This project focuses on analyzing and classifying literary characters based on the Myers-Briggs Personality Types (MBTI). The goal is to create a unique experience where users can interact with text, relate to characters based on their own personality type, and receive personalized recommendations for passages from classic English literature.

## Overview

The project uses a dataset collected from the Personality Cafe forum, consisting of approximately 8000 data rows categorized by MBTI personality types. Additionally, it includes dialogues extracted from nine classic English novels:

- 1984
- Animal Farm
- Chicago
- Death of a Salesman
- Frankenstein
- Old Man and the Sea
- Pride and Prejudice
- The Great Gatsby
- The Picture of Dorian Gray

### Key Features

![image](https://github.com/user-attachments/assets/fe743474-e2a9-4ffd-a79f-0f949b1e5f3b)


1. **Data Preprocessing**
   - Removal of URLs, special characters, and delimiters.
   - Conversion to lowercase.
   - Alignment of dialogue length to match novel excerpts.
   - Use of `RandomOverSampler` to address data imbalance.
  
2. **Classification Models**
   - **Introverted / Extroverted (I/E)**
   - **Intuitive / Observant (N/S)**
   - **Thinking / Feeling (T/F)**
   - **Judging / Prospecting (J/P)**


   Each model uses the following configuration:
   - **Loss Function**: CrossEntropy
   - **Optimizer**: AdamW
   - **Batch Size**: 16
   - **Epochs**: 10

3. **Label Mapping**
   - The project uses label mappings for various personality types to categorize characters based on the MBTI framework.

4. **Output and Recommendations**
   - Upon inputting a user’s MBTI, the system provides:
     - **Character Information**: Data is retrieved from a pre-saved database based on similarity checks.
     - **Personalized Passages**: Recommendations for the most relatable novel passages, including explanations of the scenes and their connection to the user's personality.
     - **Additional Character Suggestions**: Recommendations for other literary characters with similar traits.

## How to Use


1. **Input Your MBTI**: Start by entering your MBTI personality type.
2. **Receive Character Information**: Get detailed information about characters that match your personality.
3. **Explore Novel Passages**: Read and connect with passages from classic literature that resonate with your personality.
4. **Discover More**: Receive suggestions for other characters and novels that might interest you based on your MBTI.

## Dataset

- **Original Data**: 6940 training and 1735 validation samples (split 80:20).
- **Post-Oversampling**:
  - **Introverted/Extroverted**: 10681 training, 2671 validation samples.
  - **Intuitive/Observant**: 11964 training, 2992 validation samples.
  - **Judging/Prospecting**: 8385 training, 2097 validation samples.

## Challenges

- **Data Imbalance**: Significant imbalance observed in E/I and N/S indicators.
- **Complexity in Program Implementation**: Balancing accuracy and model complexity.

## Future Work

- **Incorporate Descriptive Text**: Extend analysis beyond dialogue to include narrative descriptions.
- **Improve J/P Reliability**: Address issues related to the J/P indicator in literary contexts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Prerequisites

![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
	![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

