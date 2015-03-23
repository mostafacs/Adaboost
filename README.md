# Adaboost

AdaBoost, short for "Adaptive Boosting", is a machine learning meta-algorithm. It can be used in conjunction with many other types of learning algorithms to improve their performance. The output of the other learning algorithms ('weak learners') is combined into a weighted sum that represents the final output of the boosted classifier
[More Information](http://en.wikipedia.org/wiki/AdaBoost)


# Implementation

I prefer looping over training file lines instead of loading all file in memory so 
it must work good with very large files.
First , Learning File must be in PSV ( Pipe Separated Values ). There is example file already exist in resources/trainingSamples.psv
Last Column work as entity Class Type  must be { 1 , -1 } "currently support binary classification only"

 
# Usage


```java
		File file = new File("resources/trainSamples.psv");
		Adaboost boosting = Adaboost.train(file, 10, 10, 0);
		int label = boosting.classify("1.|2.1".split("\\|"));
		System.out.println("Data Label[1|2.1] = "+label);
```

# Requirements
Java 8


