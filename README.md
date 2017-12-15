ud120-projects
==============

Clone the project from github:
```
$ git clone git@github.com:d-rudolf/ud120-projects.git && cd ud120-projects && git checkout project_submission 
```

The branch for project submission is project_submission, not master
The fastest way to run the code is to use pipenv.
Install pipenv:
```
$ pip install pipenv
```
I used python 3.5 throughout the project. Therefore, create a new project with python 3 interpreter:
```
$ pipenv --python 3.5 && pipenv shell
```
Install all dependencies from the Pipfile.lock. You should copy the Pipfile.lock file to the pipenv project directory. 
```
$ pipenv install --dev
```
To run the machine learning code enter
```
$ yourpath/ud120-projects/final_project$ python poi_id.py 
```
For helper functions I wrote a helper.py file, which is in the same folder as the poi_id.py file. 
I also wrote a small flask app to visualize the data. To run it locally enter
```
$ cd flask_app && python manage.py runserver 
```

Then, enter 
```
http://localhost:8080/
``` 
in the browser to use the app. The app allows to plot two features on the x- and y-axis. 
Select two buttons and press the plot button. To clear the plot press the clear button.