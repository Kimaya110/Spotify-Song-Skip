
# Using NLP Techniques to Predict Song Skips on Spotify based on Acoustic Data



## Abstract

Recommender systems are crucial in music streaming services, most notably in the form of personalised playlists. Exploring user interactions within these listening sessions can help you understand user preferences in the context of a single session. Spotify is constantly striving to improve the user experience. Using Machine Learning, we can create a model that predicts whether the user will skip the song or not. Spotify can better understand its users' needs and improve its music recommendation system by predicting whether users will skip certain songs. For this project, logistic regression and gradient boosted trees were used as models. The outcomes were obtained after training these models on a large dataset of approximately 10000 listening sessions. Our models were approximately 87% accurate.

#### Keywords: Machine Learning, Logistic Regression, Gradient Boosted Trees, Skip Prediction
## Introduction

Music consumption patterns have changed dramatically in the last decade, thanks to the rise of streaming services such as Spotify, Apple Music, and Tidal. These services are now available in a wide range of locations. These music suppliers are also rewarded for providing songs that customers enjoy in order to improve the user experience and increase time spent on their platform. As a result, determining what type of music the user wants to listen to is a significant challenge in music streaming. Because consumers can skip tracks whenever they want, the skip button is one element on these music services that plays a big role in the user experience and helps determine what a user loves. 
So the  personalized music recommendation systems are prominent in many music streaming services, such as Spotify. These recommendation systems enable users to listen to music based on a specific song, the user's mood, time, or location. The project's specific goal was to predict whether a listener would skip a particular song. The vast amount of available music, as well as the diverse interests displayed by different users, as well as by the same user in different situations, pose significant challenges to such systems.
Due to the large size of the dataset and a lack of suitable hardware, we decided to use only a sample dataset of about 10000 sessions. Each session ranges in length from 10 to 20 tracks. This means that the model must predict skipping behaviour for five tracks in the shortest sessions and ten tracks in the longest sessions. For each track, metadata such as duration, release year, and estimated US popularity are provided. There are also audio features such as acousticness, tempo, and loudness. Interactions such as seek forward/backwards, short/long pause before play are available for each track presented to the user during the session. Finally, session data such as the time of day.
Finally, session information such as the time of day, date, whether or not the user is a premium user, and the context type of playlist are present. The dataset categorises skipping behaviour into four types: (1) skip 1: Boolean indicating whether the track was only played briefly (2) skip 2: Boolean indicating whether the track was only played briefly (3) skip 3: Boolean indicating whether the track was played in its entirety (4) not skipped: Boolean indicating that the track was played in its entirety The challenge restricts itself to predicting only the skip 2 behaviour.
## Methodology

We used a simple methodology used in most machine learning projects for this project. All of the steps we took are outlined below:
## Data Gathering

Spotify's dataset is divided into two parts: session logs and tracks. Spotify's dataset contains approximately 130 million listening sessions. We chose to use a sample of the dataset due to a lack of suitable hardware. There are 10000 listening sessions in the session logs. Each of these listening sessions is defined by 21 characteristics, including:
unique session ID,
the sequence of tracks played during the session,
the date and hour the session was played, etc.

The tracks dataset contains 50704 distinct tracks that users heard during their listening sessions. Each of these tracks is defined by 29 features, including:
unique track ID,
track duration,
track popularity rating,
track beat strength etc.



## Data Pre-processing

Pre-processing data is an essential step in any machine learning project. Data pre-processing is the process of dealing with missing values, parsing dates, handling inconsistent data, and so on. Because there were no missing values in our case, dealing with missing values was unnecessary. The dataset was provided in two files. So we started by merging the datasets based on track id. Because all of the dates were from the same year but in different months, which was converted from years to days.
## Data Visualization

Data visualisation is commonly used to gain insight into the data. We made histograms and bar plots of the features to see how they differed depending on the output feature. We also used a built-in Python library called sweetviz, which does all of the visualisation for us in a beautiful format.
## Exploratory Data Analysis (EDA)

EDA is unavoidable and one of the most important steps in fine-tuning the given data sets in a different form of analysis to understand the insights of the key characteristics of various entities of the data set like columns, rows by using Pandas, NumPy, Statistical Methods, and Data visualisation packages. We checked for correlation in EDA and eliminated features that were highly positively and highly negatively correlated. We then used one-shot encoding to encode the categorical features. Because a machine learning model can only work with numerical data, we require encoding. Following that, we looked for and removed outliers from the dataset. We scaled the dataset using a min-m method after removing the outliers. We scaled the dataset after removing the outliers using a min-max scaler, which scales the dataset into the range 0 to 1.
## Train-Test Split

The train-test split divides the dataset into training and testing so that after training the model, we can evaluate its accuracy on new values that it has never seen before. We used 80% of the data for training and the remaining 20% for testing.
## Model Selection

The task, according to the project guidelines, was to create models using Logistic Regression and Light Gradient Boosted Trees. I began with Logistic Regression because it is simple to implement and works well for classification problems. The accuracy I obtained with Logistic Regression was approximately 87 percent, which was quite good considering we were only working on a subset of the entire dataset. Following that, I moved on to Light Gradient Boosted Trees, a tree-based model that employs the boosting Technique.
I was able to achieve an accuracy of 87.76 percent after fine-tuning the model's hyperparameters. The accuracy I obtained from both models was nearly identical (LBGT winning with a small margin). However, because LGBT is a very robust algorithm with many parameters that can be tuned, there is always room for improvement. I also used the confusion matrix to determine how many features were incorrectly classified. Finally, I used the pickle library to save the models to a file.
## Model Deployment

For deployment, I used Flask, a Python web development framework. I created a simple frontend for my web app using HTML and CSS. By providing the track name, I created a class that would extract the track-ID and track features from the Spotify API and pass them to the model. To host my web app, I used Heroku, a cloud platform.
## Conclusion and Future Work

Creating a sequential skip prediction model was a difficult task. At the same time, these types of projects are extremely beneficial to companies that provide music services in terms of developing recommendation systems. The accuracy I obtained is approximately 87%, which I believe is quite good given the time and hardware constraints. Also, I am confident that the accuracy would have been much higher if we had used the entire dataset provided by Spotify rather than a sample of it.

In future, we may be able to use models based on Neural Networks to accomplish this task. We can use LSTM and Bi-LSTM models because they have excellent long-term memory. We can also use the Encoder-Decoder architecture to keep the data sequence. Transformers, which are a very complicated model but are very good for these kinds of sequential predictions, could also be used.