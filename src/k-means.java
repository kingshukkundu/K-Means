// ============================================================================
//   Copyright 2019 Kingshuk Kundu
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
// ============================================================================

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import java.lang.Math;

/**
 * Encapsulates a K-Means algorithm.  Exports the clustered dataset
 * based on k-means in a separate excel file and prints the execution time 
 * and number of iterations on the console
 * @author Kingshuk Kundu
 */

public class KMeans {

	public static void main(String[]args) throws IOException 
	{	
		Scanner dt = new Scanner(System.in);
		System.out.print("Enter the name of the CSV file: ");
		String fileName = dt.nextLine();
		
		double dataset[][]= importcsv(fileName+".csv");
		//double dataset[][]= importcsv("C:\\\\Users\\\\[user_name]\\\\[location]\\\\[file_name].csv");  

		System.out.println("Choose the number of clusters (>=2) :");

		Scanner in = new Scanner(System.in);
		int NumOfClstr= in.nextInt();

		System.out.println("Please choose the maximum number of iterations: ");

		int MaxIt=in.nextInt();

		

		ArrayList<double[]> cluster = new ArrayList<double[]>();
		ArrayList<ArrayList<double[]>> masterclusters  = new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<double[]>> oldcluster =  new ArrayList<ArrayList<double[]>>();

		for (int a=0; a<NumOfClstr; a++) {

			cluster= new ArrayList<double[]>(cluster);
			masterclusters.add(cluster);

			cluster= new ArrayList<double[]>(cluster);
			oldcluster.add(cluster);

		}

		ArrayList<double[]> centroid=new ArrayList<double[]>();
		System.out.println("\nPlease pick a number between 1 and "+dataset.length+"\n\n");

		for(int b=1; b<NumOfClstr+1; b++) {

			System.out.println("Centroid number "+b+": Pick the row number of the centroid column in the dataset:");

			Scanner sc = new Scanner(System.in);
			int input = sc.nextInt(); 

			centroid.add(dataset[input-1]);

			if(b==NumOfClstr) {

				sc.close();

			}

		}

		in.close();
		dt.close();
		
		int clstrno;
		double tempdst;

		for(int c=0;c<dataset.length;c++) {

			clstrno=0;
			tempdst=CalcDist(dataset[c],centroid.get(0));

			for(int d=0;d<centroid.size();d++) {

				double temp= CalcDist(dataset[c],centroid.get(d));

				if (temp<tempdst){

					tempdst=temp;
					clstrno=d;
				}

			}

			masterclusters.get(clstrno).add(dataset[c]);

		}

		for(int e=0; e<centroid.size();e++) {

			centroid.set(e,CalcNewCentroid(ConvToArr(masterclusters.get(e))));

		}

		oldcluster=resetcluster(oldcluster,masterclusters);
		int iterations=0;
		long startTime = System.nanoTime();

		while(true) {

			for (int g=0; g<masterclusters.size();g++) {

				for(int h=0; h<masterclusters.get(g).size();h++) {

					clstrno=0;
					tempdst=CalcDist(masterclusters.get(g).get(h),centroid.get(0));

					for(int d=0;d<centroid.size();d++) {

						double temp= CalcDist(masterclusters.get(g).get(h),centroid.get(d));

						if (temp<tempdst){

							tempdst=temp;
							clstrno=d;

						}

					}

					double store[] =masterclusters.get(g).get(h);
					masterclusters.get(g).remove(h);
					masterclusters.get(clstrno).add(store);

				}

				for(int e=0; e<centroid.size();e++) {

					centroid.set(e,CalcNewCentroid(ConvToArr(masterclusters.get(e))));
				}

			}

			print2D(ConvToArr(centroid));
			iterations++;

			if(iterations >= MaxIt || checkEquality(oldcluster, masterclusters))
				break;
			else
				oldcluster=resetcluster(oldcluster, masterclusters);

		}

		long endTime   = System.nanoTime();
		double totalTime = endTime - startTime;
		double exectime= totalTime/1000000000;

		System.out.println("Total execution time: "+exectime+" seconds");
		System.out.println("Total number of iteration: "+iterations);

		writecsv(masterclusters);	

	}

	//imports CSV as a double[][]
	public static double[][] importcsv(String loc) throws IOException
	{	
		String thisLine;
		BufferedReader br = new BufferedReader(new FileReader(loc));
		ArrayList<String[]> lines = new ArrayList<String[]>();

		while ((thisLine = br.readLine()) != null) {

			lines.add(thisLine.split(","));

		}

		br.close();	

		ArrayList<double[]> result =new ArrayList<double[]>();	

		for(int i=0; i<lines.size();i++) {

			double[] temp = new double [lines.get(i).length];

			for(int k=0; k<lines.get(i).length;k++) {

				temp[k]= Double.parseDouble(lines.get(i)[k]);

			}

			result.add(temp);

		}

		return ConvToArr(result);
	}

	//calculate distance from a centroid based on values from one line of dataset 
	public static double CalcDist(double arr[], double centroid[])
	{

		double d=0;

		for(int i=1;i<arr.length;i++ ) {

			double l= arr[i];	//skipping the first line to store ID/row number
			d=d+Math.pow((l-centroid[i]),2);	

		}

		d=Math.sqrt(d);
		return d;
	}

	//calculate new Centroids based on nearest data points to previous Centroid which was input as 2D array
	public static double[] CalcNewCentroid(double closestpts[][]) 
	{
		double centroid[] = new double[closestpts[0].length];	

		for (int j=1; j<closestpts[0].length; j++) { //skipping first column because its an id

			double temp=0; 

			for (int i=0; i<closestpts.length;i++) {

				temp+=closestpts[i][j];
			}

			temp=temp/closestpts.length;
			centroid[j]=temp;

		}

		return centroid;
	}

	// converts ArrayList of array of double to a 2D Array
	public static double[][] ConvToArr(ArrayList<double[]> input)
	{	
		double[][] arr = new double[input.size()][input.get(0).length];

		for(int i=0; i<input.size(); i++){

			arr[i] = input.get(i);	
		}

		return arr;
	}

	//Checks if old clusters are equal to new clusters
	static boolean checkEquality(ArrayList<ArrayList<double[]>> oldClusters, ArrayList<ArrayList<double[]>> newClusters)
	{
		for(int i=0; i<newClusters.size(); i++) {

			if(oldClusters.size() != newClusters.size())

				return false;

			for(int j=0; j<newClusters.get(i).size(); j++) {

				if(oldClusters.get(i).size() != newClusters.get(i).size()) {

					return false;

				}
				for(int v=0;v<newClusters.get(i).get(j).length;v++ ) {

					if(oldClusters.get(i).get(j)[v] != newClusters.get(i).get(j)[v]) {

						return false;

					}

				}

			}

		}

		return true;

	}

	//sets old clusters equal to new clusters
	static ArrayList<ArrayList<double[]>> resetcluster(ArrayList<ArrayList<double[]>> oldClusters, ArrayList<ArrayList<double[]>> newClusters) 
	{
		for(int i=0; i<oldClusters.size(); i++) {	

			oldClusters.get(i).clear();

			for(int j=0; j<newClusters.get(i).size(); j++) {

				double[] Array1D= new double [newClusters.get(i).get(j).length] ;

				for(int v=0;v<newClusters.get(i).get(j).length;v++ ) {

					Array1D[v]=newClusters.get(i).get(j)[v];
				}

				oldClusters.get(i).add(Array1D);
			}

		}

		return oldClusters;

	}

	public static void print2D(double mat[][]) 
	{ 
		// Loop through all rows 
		for (double[] row : mat) {

			// Loop through all columns of current row 
			for (double x : row) {

				System.out.print(x + " "); 
			}

			System.out.println();
		}

		System.out.println();

	}

	//writes the ArrayList of ArrayList of array of doubles to a CSV file
	public static void writecsv(ArrayList<ArrayList<double[]>>fresult) throws IOException 
	{
		FileWriter csvWriter = new FileWriter("new.csv");
		int cluster=1;
		
		for (ArrayList<double[]> clusterdata : fresult) {
			
			for(double[] row:clusterdata) {
				
				csvWriter.append(cluster+",");
				
				for(double d:row) {

					csvWriter.append(String.valueOf(d));
					csvWriter.append(",");

				}

				csvWriter.append("\n");

			}

			csvWriter.append("\n\n\n\n\n");	
			cluster++;
			
		}

		csvWriter.close();

	}

}
