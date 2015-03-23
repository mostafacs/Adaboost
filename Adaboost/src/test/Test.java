package test;

import java.io.File;
import java.io.IOException;

import com.classifiers.boosting.Adaboost;

public class Test {

	public static void main(String[] args) {

		File file = new File("resources/trainSamples.psv");
		Adaboost boosting;
		try {
			boosting = Adaboost.train(file, 10, 10, 0);
			int label = boosting.classify("1.|2.1".split("\\|"));
			System.out.println("Data-Label[1|2.1] = " + label);

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
