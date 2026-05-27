# Final-Year-Project- Machine Generated vs Human Generated Text Detection

In this project we seek create a model which can detect and flag machine generated text. For this, we have used data made available from various resources like Kaggle, GitHub as well as collecting responses after giving prompts to LLM models like GPT and BARD. 

Machine Generated Text is becoming increasingly common in a variety of applications, such as news articles, social media posts, and product descriptions. However, Machine Generated Text can also be used for malicious purposes, such as spreading misinformation or propaganda.

In this minor project we have developed a machine learning model to detect Machine Generated Text. The model is trained on a dataset of labeled text, including both Machine Generated Text and human-generated text. The model will then be used to predict whether a new piece of text is Machine Generated Text or not.

The project uses a variety of machine learning techniques, such as natural language processing (NLP) and deep learning. The NLP techniques will be used to extract features from the text, such as the length of the sentences, the complexity of the vocabulary, and the presence of certain patterns. The deep learning techniques will be used to train a model to predict whether the text is Machine generated Text or not based on the extracted features.

## Introduction

- Over the past few years, machine-generated text has risen to popularity and received heavy improvements in accuracy and efficiency. Nowadays it is safe to say that it is beyond the scope of human intelligence to detect the minute differences between text that is human-generated and text that a machine has created. However, this potent technology also introduces potential misuse and societal implications, including privacy violations, misinformation, and integrity and originality challenges in academia.

- In an era characterized by an ever-expanding digital landscape and an exponential increase in textual data, the ability to discern between human-generated and machine-generated text has become a critical challenge. The phenomenon of generating text through artificial intelligence (AI) models has proliferated across various domains, from content generation and chatbots to misinformation and deepfake production. As a result, the need for robust and reliable "Machine Generated Text Detection" has emerged as a pressing concern in both technological and societal contexts.

- This minor project aims to delve into the intricate world of identifying machine-generated text amidst the vast ocean of digital content. The machine-generated text encompasses a wide spectrum of applications, ranging from algorithmically produced news articles to automated social media posts and spam emails. Detecting such text is vital for safeguarding the integrity of information dissemination, maintaining trust in online interactions, and mitigating the risks associated with AI-generated content, such as misinformation and cyber threats.

- The objective of this project is to develop an efficient and accurate text detection system that can distinguish between human and machine-generated text. This system will employ cutting-edge natural language processing (NLP) techniques, machine
learning algorithms, and deep neural networks to analyze and classify text data. By doing so, it will contribute to enhancing the digital ecosystem's authenticity and security.

## Built with 🛠️
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png"></code>
<code><img height="30" src="https://github.com/LakshayGupta11/Minor-Project--Machine-Generated-vs-Human-Generated-Text-Detection/assets/140741956/cb3db2a7-2478-4770-9e42-29b0244f68d2"></code>
<code><img height="30" src="https://raw.githubusercontent.com/numpy/numpy/7e7f4adab814b223f7f917369a72757cd28b10cb/branding/icons/numpylogo.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg"></code>
<code><img height="30" src="https://matplotlib.org/_static/logo2.svg"></code>
<code><img height="30" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1280px-Scikit_learn_logo_small.svg.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pytorch/pytorch/39fa0b5d0a3b966a50dcd90b26e6c36942705d6d/docs/source/_static/img/pytorch-logo-dark.svg"></code>
<code><img height="30" src="https://github.com/LakshayGupta11/Minor-Project--Machine-Generated-vs-Human-Generated-Text-Detection/assets/140741956/2871ea02-c557-4b05-b0cd-a54b19419ce5"></code>
<code><img height="30" src="https://github.com/LakshayGupta11/Minor-Project--Machine-Generated-vs-Human-Generated-Text-Detection/assets/140741956/dccd0736-90d8-4c4a-91b7-a4ae6ba5faf8"></code>
<code><img height="30" src="https://github.com/LakshayGupta11/Minor-Project--Machine-Generated-vs-Human-Generated-Text-Detection/assets/140741956/ad5eba75-952d-4567-8fa5-8e1adeea3c6d"></code>

## Layers Used Throughout the Project

- Input Layer:
The Input layer in a neural network serves as the entry point for data and defines the shape of the input data. In the context of text classification:

Text Data Representation:
For text-based tasks, the Input layer receives the input text data, often represented as sequences of words, tokens, or embeddings.

Input Shape Definition:
The Input layer defines the shape of the input data. In NLP tasks, this might involve specifying the length of input sequences or the dimensionality of word embeddings.

Data Preparation:
Before feeding text data into the network, it needs to be encoded or tokenized into numerical representations suitable for the model. The Input layer accommodates these preprocessed numerical inputs.

Example:
In the case of a Convolutional Neural Network (CNN) for text classification, the Input layer may receive preprocessed text data in the form of word embeddings or tokenized sequences.

- Dense Layer:
The Dense layer, also known as a fully connected layer, is a fundamental neural network layer that connects each neuron in the current layer to every neuron in the subsequent layer.

Feature Learning:
In the context of text classification, Dense layers aid in learning higher-level features from the input representations (e.g., word embeddings or encoded text sequences).

Non-linearity:
Dense layers apply non-linear transformations to the input data, enabling the network to capture complex patterns and relationships within the text data.

Output Transformation:
The Dense layer transforms the input from the previous layer into an output suitable for the subsequent layers, often by applying weights and biases and using activation functions (e.g., ReLU, Sigmoid, or Softmax).

Example:
After processing input text data through earlier layers (e.g., embedding or convolutional layers), the Flatten layer reshapes the data into a one-dimensional vector before passing it to a Dense layer. The Dense layer processes this vector and applies weights to compute outputs.

- Flatten Layer:
The Flatten layer is used to convert multi-dimensional input data into a one-dimensional form, allowing it to be fed into a Dense layer.

Dimensionality Reduction:
In the context of neural networks for text, the Flatten layer is used to transform output data from convolutional or recurrent layers (which have multi-dimensional outputs) into a flat vector form.

Transition to Fully Connected Layers:
Neural network architectures often transition from convolutional or recurrent layers to Dense layers for classification or prediction. The Flatten layer facilitates this transition by converting the 2D or 3D output into a 1D vector.

Example:
In CNN architectures for text classification, after applying convolutional layers for feature extraction from text data, a Flatten layer is used to flatten the output feature maps into a one-dimensional vector before passing it to Dense layers for classification.

- Conv1D Layer:
The Conv1D layer is a one-dimensional convolutional layer used to perform convolutions on one-dimensional input data, such as sequences of text. Here's how it's used in our project:

Feature Extraction:
For text data, Conv1D layers apply filters/kernels across the input text sequences, allowing the model to extract local patterns and features, such as combinations of words or phrases that are relevant for classification.

Learning Local Patterns:
Conv1D layers learn representations by sliding a 1D filter/kernel across the input sequences, detecting patterns and capturing spatial relationships between words or tokens.

Example:
In our text detection model, the Conv1D layer could process the word embeddings or tokenized sequences of text, extracting features like combinations of words or phrases that distinguish between human and machine-generated text.

- Dropout Layer:
The Dropout layer is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to zero during training. Here's how it's used in our project:

Reducing Overfitting:
Dropout layers randomly drop a certain proportion of neurons during each training iteration, forcing the model to learn redundant representations and preventing reliance on specific features. This helps in generalizing better to unseen data.

Improving Model Robustness:
By removing some connections temporarily, Dropout encourages the network to learn more robust features, reducing the model's sensitivity to noise or specific patterns in the training data.

Example:
In our text detection model, applying Dropout after Conv1D or Dense layers helps in preventing the model from overfitting to the training data and improves its ability to generalize to unseen text samples.

- Batch Normalization Layer:
Batch Normalization is a technique used to improve the stability and speed of training deep neural networks by normalizing the inputs of each layer. Here is how it is used in our project:

Normalization of Activations:
Batch Normalization normalizes the activations of each layer by subtracting the batch mean and dividing by the batch standard deviation. This helps in stabilizing the training process and reduces internal covariate shift.

Faster Convergence:
By normalizing the activations, Batch Normalization enables higher learning rates, leading to faster convergence during training. It also reduces the dependence of gradients on the scale of parameters, aiding in smoother optimization.

Example:
In our text detection model, Batch Normalization layers could be placed after Conv1D or Dense layers, normalizing the activations before applying activation functions, thus improving the stability and convergence speed of the network.

- Concatenate Layer:
The Concatenate layer in neural networks concatenates (joins together) the outputs from multiple layers along a specific axis. Here is how it might be used in our project:

Combining Features:
The Concatenate layer combines the outputs from different layers, enabling the network to consider information from various parts of the network and different representations.

Feature Fusion:
n text-related tasks, Concatenate layers can merge features extracted by different layers (e.g., Convolutional and LSTM layers), allowing the model to capture diverse aspects and representations of the text data.

Example:
In our text detection model, the Concatenate layer might combine the outputs from different branches or different types of layers (e.g., outputs from Conv1D and LSTM layers) to capture both local and sequential information from text sequences.

- LSTM Layer:
LSTM is a type of recurrent neural network (RNN) that is well-suited for processing sequential data due to its ability to capture long-term dependencies. Here is how it is used in our project:

Sequential Informational Handling:
LSTM layers are effective in processing sequences by maintaining an internal state and selectively remembering or forgetting information over long sequences, making them suitable for handling text data.

Long-Term Dependencies:
Unlike traditional RNNs, LSTMs mitigate the vanishing gradient problem and can capture long-range dependencies in text sequences, allowing the model to retain information over extended contexts.

Example:
In our text detection model, LSTM layers can process the sequential nature of text data, analyzing the relationships between words across sentences or paragraphs, which might be crucial for differentiating between human and machine-generated text.

- Max-Pooling Layer:
The MaxPooling layer is a pooling operation that reduces the spatial dimensions of the input by taking the maximum value within a fixed kernel size. Here is its relevance in our project:

Down sampling:
MaxPooling layers reduce the dimensionality of feature maps, focusing on the most important information by retaining the maximum values within specific regions.

Feature Reduction:
By discarding non-maximal values, MaxPooling layers retain the most relevant features, aiding in reducing computational complexity and preventing overfitting.

Example:
In our text detection model, MaxPooling layers might be used after Conv1D layers to reduce the spatial dimensionality of the feature maps obtained from text representations, retaining the most relevant information for classification tasks.

- Average Layer:
The Average layer computes the average value across a specific axis of input data. Here is how it is utilized in our project:

Pooling and Aggregation:
The Average layer aggregates information by computing the mean value across a dimension, summarizing information, and reducing dimensionality.

Global Representation:
In text-related tasks, Average layers can summarize or represent text features globally by computing the average of embedded or transformed representations, providing an overall view of the text.

Example:
In our text detection model, the Average layer can be applied after obtaining embeddings or hidden representations, aggregating information across the sequence to obtain a global representation of text before making predictions.

## Archtecture of the model

<img width="284" alt="arc" src="https://github.com/LakshayGupta11/Minor-Project--Machine-Generated-vs-Human-Generated-Text-Detection/assets/140741956/fbbaef56-6152-4945-a0c8-259594ca2265">

## Applications

- Understanding the limitations of machine-generated text: Machine-generated text, such as that produced by language models like GPT-3, can be difficult to distinguish from human-written text. By developing a machine-generated text detection system, we gained insights into the strengths and weaknesses of these models and better understand their potential impact on various domains, such as journalism, content creation, and social media.

- Improving the trustworthiness of online content: With the proliferation of machine-generated text, it is becoming increasingly important to ensure the trustworthiness of online content. By detecting and flagging machine-generated text, we can help users identify potentially unreliable or misleading information and make more informed decisions.

- Enhancing cybersecurity and threat detection: Machine-generated text can be used in various cyber-attacks, such as phishing, social engineering, and disinformation campaigns. By developing a machine-generated text detection system, we can contribute to the field of cybersecurity and help organizations better protect themselves against these threats.

- Assisting content moderators and fact-checkers: Content moderation and fact-checking are time-consuming tasks, especially when dealing with large volumes of text. By automating the detection of machine-generated text, we can help content moderators and fact-checkers prioritize their work and focus on more critical tasks, such as verifying the accuracy of human-written content.

- Supporting research on machine-generated text: As machine-generated text continues to advance, there is a need for more research on its impact and potential applications. By developing a machine-generated text detection system, we can contribute to this research and help advance our understanding of the capabilities and limitations of these models.

## Results

For the bellow article, the model was tested and it showed the following results:
(https://www.theguardian.com/australia-news/2024/jan/22/australian-wildlife-threatened-species-list-record-added-2023)

![test 1](https://github.com/LakshayGupta11/Minor-Project--Machine-Generated-vs-Human-Generated-Text-Detection/assets/140741956/838c22fe-83c9-4b37-807e-77235b84ea00)

whereas when this issue was given as ChatGpt prompt the following results were obtained:

![test 2](https://github.com/LakshayGupta11/Minor-Project--Machine-Generated-vs-Human-Generated-Text-Detection/assets/140741956/89dbde28-351a-4a8e-bac2-a7bf7db94f94)



