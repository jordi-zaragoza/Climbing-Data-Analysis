# Climbing Data Analysis
Analyzed climbing data as a project for my IronHack course

## Overview
I used Python3 to explore the data of the bigest climbing website: 8a.nu 

I Tried to answer some questions like:
- What is the role of the hight-weight rate on the max grade
- Is there any difference between women-men mean grade?
- What are the most climbed routes worldwide
- Where to go climbing on Summer-Winter

I also did a model able to predict the expected level of a climber using his/her features. A model that can be also well improved, I welcome anyone who wants to join me in this task :)


## Contents
0. `tableau/Project_clean.ipynb` - This is the presentation in tableau
1. `src/1.Project_clean.ipynb` - This module takes the 3 main databases: Ascent, Users, Grades. Combines and clean them in order to get the 3 main tables used for the presentation
2. `src/2.Project_transform.ipynb` - This module applies the transformations to the main dataframe and saves the transformers on the `/transformer` folder using pickle
3. `src/3.Project_model.ipynb` - This module creates 3 different models: linear, KNN and Neural Networks.
4. `src/4.Project_use_the_model.ipynb` - This module uses the best model generated to predict the expected grade of a climber and also the expected evolution during the following years
5. `src/5.Statistical_Comparatives.ipynb` - This module does a mean comparative between men-women using the Aspin-Welch  t-test method

### Databases
The databases are not stored in my git but you can find them in this link: https://www.kaggle.com/dcohen21/8anu-climbing-logbook


### Requirements
1. Python3 and standard libraries
2. The following Python3 libraries:
    1. matplotlib==3.3.4
    2. numpy==1.19.5
    3. pandas==1.4.1
    4. scikit_learn==1.0.2
    5. scipy==1.6.0
    6. seaborn==0.11.2


### Acknowledgments
Thanks to David Cohen for creating the scrapper (https://github.com/mplaine/8a.nu-Scraper) and hosting the data on Kaggle website (https://www.kaggle.com/dcohen21/8anu-climbing-logbook)


### Future Directions
As David is mentioning in his Scrapper, I also it would be a nice idea to build a route-recommendation engine based on this data. If you're interested in collaborating or have any ideas you want to share, message him. If I get any idea I will do the same.




