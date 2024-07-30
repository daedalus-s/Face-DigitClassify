from subprocess import call
def output():
    classifier = ['naiveBayes', 'perceptron', 'kNearestNeighbors' ]
    #classifier = ['kNearestNeighbors']
    data_type = ['digits','faces']
    for i in data_type:
        for j in classifier:
            print("")
            print("Running : ", j + " -> " + i)
            call(["python","dataClassifier.py",j,i])

output()    
