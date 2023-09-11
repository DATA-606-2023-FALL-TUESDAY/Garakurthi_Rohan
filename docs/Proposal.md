## 1. Title and Author

- Vehicle Type Detection for Intellignet Transportation Systems & Traffic Management
- Rohan Garakurthi
- https://github.com/Rohan198/UMBC-DATA606-FALL2023-TUESDAY
- https://www.linkedin.com/feed/?trk=guest_homepage-basic_google-one-tap-submit
- https://docs.google.com/presentation/d/1BZiTZgs6JVvtrBKOSgS4Ys6amEEv9bCi/edit?usp=drive_link&ouid=112142881428497726904&rtpof=true&sd=true
    
## 2. Background
In today's rapidly evolving world, intelligent transportation systems are essential for the smooth and efficient functioning of our cities and economies. As vehicles of all shapes and sizes traverse our road networks, it becomes increasingly crucial to have automated systems that can classify and identify these vehicles accurately. 

This project "Vehicle Type Detection" aims to provide a versatile solution for a multitude of applications by leveraging the power of deep learning and computer vision to analyze images and automatically categorize vehicles into distinct types. 

From improving traffic management and surveillance to enhancing security and supporting autonomous vehicles, the implications of accurate vehicle type detection are vast. We explore the intricacies of identifying vehicles from images, ultimately contributing to smarter, safer, and more efficient transportation systems.

Some practical application where vehicle type detection can be of use:
●	Road Safety, Traffic Management and Surveillance
●	Smart Parking Solutions, Toll Collection and Access Control
●	Vehicle Insurance Risk Assessment

Research Question(s):
1.	How accurately simple CNN models with less layers can classify this data?
2.	What parts of images are concentrated by the models during classification?
3.	How pretrained models based on deep CNN layers compare in classifying vehicle types?
4.	How different data augmentation techniques can impact the model performance?
5.	How accurately can different model(s) classify the data?


## 3. Data 

Describe the datasets you are using to answer your research questions.

Link: https://www.kaggle.com/datasets/kaggleashwin/vehicle-type-recognition

●	Number of Images: 400
●	Number of Vehicles types/classes : 4
●	Datasize: =~ 200 MB

The dataset comprises a diverse collection of images for four distinct classes (100 of each class): Car, Truck, Bus, and Motorcycle. 

More data collection: While the size of this dataset is small, based on initial results from the model training & evaluation part, I plan to extend the size of the data set with help of web scraping or bringing in the data from other datasets (like Open Images dataset).

## 4. Exploratory Data Analysis (EDA)

- Perform data exploration using Jupyter Notebook
- You would focus on the target variable and the selected features and drop all other columns.
- produce summary statistics of key variables
- Create visualizations (I recommend using **Plotly Express**)
- Find out if the data require cleansing:
  - Missing values?
  - Duplicate rows? 
- Find out if the data require splitting, merging, pivoting, melting, etc.
- Find out if you need to bring in other data sources to augment your data.
  - For example, population, socioeconomic data from Census may be helpful.
- For textual data, you will pre-process (normalize, remove stopwords, tokenize) them before you can analyze them in predictive analysis/machine learning.
- Make sure the resulting dataset need to be "tidy":
  - each row represent one observation (ideally one unique entity/subject).
  - each columm represents one unique property of that entity. 

## 5. Model Training 

- What models you will be using for predictive analytics?
- How will you train the models?
  - Train vs test split (80/20, 70/30, etc.)
  - Python packages to be used (scikit-learn, NLTK, spaCy, etc.)
  - The development environments (your laptop, Google CoLab, GitHub CodeSpaces, etc.)
- How will you measure and compare the performance of the models?

## 6. Application of the Trained Models

Develop a web app for people to interact with your trained models. Potential tools for web app development:

- **Streamlit** (recommended for its simplicity and ease to learn)
- Dash
- Flask

## 7. Conclusion

- Summarize your work and its potetial application
- Point out the limitations of your work
- Lessons learned 
- Talk about future research direction

## 8. References 

List articles, blogs, and websites that you have referenced or used in your project.
