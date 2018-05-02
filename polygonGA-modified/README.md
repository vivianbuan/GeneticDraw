# polygonGA
Using Genetic Algorithm based techniques to generate images resembling a target image using random polygons. 

I've implemented two algorithms, a full fledged GA which involves Selection, Crossover, Mutation and Elitism among several members of a sample population and a simple algorithm which mutates a single Genotype in each iteration and selects it if it is 'fitter' than our present Genotype. Produces some acceptable results for simpler pictures. Fails splendidly for others. 


Steps to run:
1. pip install -r requirements.txt 
2. Specify the input image and other global parameters in main.py and helper_classes.py
3. python main.py

P.S.- It is advisable to run it on Amazon EC2 or Heroku. I've added support to link the script with your Dropbox account for saving images. Dropbox API requires an access token. 
