import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;


import java.lang.Object;
/*This program is the implementation of Hybrid Intelligent Sampling Technique Called SSOmaj-SMOTE-SSOmin

If you want to use this in the research work, please site the paper below:

Susan, Seba, and Amitesh Kumar. "SSOMaj-SMOTE-SSOMin: Three-step intelligent pruning 
of majority and minority samples for learning from imbalanced datasets."
Applied Soft Computing 78 (2019): 141-149.

link: https://www.sciencedirect.com/science/article/pii/S1568494619300924

Caution: only files with *.arff extensions are used in this program

Method of evaluation:  We have used Area Under the Curve (AUC) for evaluation purposse

You only need to provide the path to directory where the data file is stored. The input file needed to be stored in such directory.
The output will be generated in same directory where the input file is stored.

*/
public class SSOmaj_SMOTE_SSOmin {

	public Instances training_set;
	public Instances testing_set;
	public Instances In_training_set;
	public Instances In_testig_set;
	public String base_classifier="j48";
	public ArrayList<Instances> optimum_sets;
	public int iteration_number=100;
	public int population_size=100;
	private final double w=0.689343;
	private final double c1 = 1.42694;
	private final double c2 = c1;
	private final double max_velocity = 0.9820; 
	private final double min_velocity = 0.0180;
	
	private int[][] particles;
	private double[] localBest;
	private double globalBest;
	private double[][] velocity;
	private int[][] localBestParticles;
	private int[] globalBestParticles;
	
	private Random random;
	private double avgFitness;
	private DecimalFormat dec = new DecimalFormat("##.####");
	
	private Instances fitness_train_set;
	private Instances fitness_test_set;
	
	private int minority_class_size;
	public Instances balance_set;
	Hashtable<String, Integer> hash_table;
	private int max_count=0;
	
	
	SSOmaj_SMOTE_SSOmin(String directory_name, String file_name)
	{
		this.optimum_sets=new ArrayList<Instances>();
		this.random=new Random(System.currentTimeMillis());
		this.avgFitness=0.0;
		
		try {
			this.training_set=new Instances(new BufferedReader(new FileReader(directory_name+file_name)));
			this.training_set.setClassIndex(this.training_set.numAttributes()-1);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.out.println("Error !!! in retriving file");
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("Error !!! in retriving file");
		}
	}
	
	public void parameters_details()
	{
		System.out.println("-----------------Parameters Details------------------");
		System.out.println("w = "+this.w);
		System.out.println("c1 = "+this.c1);
		System.out.println("c2 = "+this.c2);
		System.out.println("Iteration number = "+this.iteration_number);
		System.out.println("Popultion number = "+this.population_size);
		System.out.println("Base Classifier = "+this.base_classifier);
	}
	
	public Instances getTraining_set() {
		return training_set;
	}

	public void setTraining_set(Instances training_set) {
		this.training_set = training_set;
	}

	public static void main(String[] args) {
	         
		
		System.out.println("This program is the implementation of Hybrid Intelligent Sampling Technique Called SSOmaj-SMOTE-SSOmin");
		System.out.println("\nIf you want to use this in the research work, please site the paper below: \n");
		System.out.println("Susan, Seba, and Amitesh Kumar. \"SSOMaj-SMOTE-SSOMin: Three-step intelligent pruning ");
		System.out.println("of majority and minority samples for learning from imbalanced datasets.\"");
		System.out.println("Applied Soft Computing 78 (2019): 141-149. \n");
		System.out.println("\nlink: https://www.sciencedirect.com/science/article/pii/S1568494619300924");
		System.out.println("\nCaution: only files with *.arff extensions are used in this program");
		System.out.println("\nMethod of evaluation:  We have used Area Under the Curve (AUC) for evaluation purposse");
		System.out.println("\nYou only need to provide the path to directory where the data file is stored. The input file needed to be stored in such directory.");
		System.out.println("\nThe output will be generated in same directory where the input file is stored.");
		System.out.println("\n=====================================================================================================================================================");
		Scanner in = new Scanner(System.in);
		System.out.println("Enter directory path name : ");
		String directory_name = in.nextLine(); 
		System.out.println("Enter file name : ");
		String file_name = in.nextLine();
		in.close();
		
		
		SSOmaj_SMOTE_SSOmin object_1=new SSOmaj_SMOTE_SSOmin(directory_name, file_name);
		object_1.parameters_details();
		
		object_1.setTraining_set(object_1.training_set);
		Instances copy_training_set=object_1.getTraining_set();
		copy_training_set.stratify(2);
		
		for(int fold=0;fold<2;fold++)
		{
			//division of data set into training and testing data set
			object_1.In_training_set=copy_training_set.trainCV(2, fold);
			Instances test=copy_training_set.testCV(2, fold);
			object_1.In_testig_set=new Instances(test);
			
			Instances in_train_minor=new Instances(object_1.In_training_set);
			in_train_minor.delete();
			Instances in_train_major=new Instances(object_1.In_training_set);
		    in_train_major.delete();
		    
		    for(int i=0;i<object_1.In_training_set.numInstances();i++)
		    {
		    	Instance instance=object_1.In_training_set.instance(i);
		    	if(instance.classValue()==1)
		    	{
		    		in_train_minor.add(instance);
		    	}
		    	else
		    	{
		    		in_train_major.add(instance);
		    	}
		    }
		    
		    for(int i=0;i<test.numInstances();i++)
		    {
		    	Instance instance=test.instance(i);
		    	if(instance.classValue()==1)
		    	{
		    		object_1.In_testig_set.add(instance);
		    		object_1.In_testig_set.add(instance);
		    	}
		    }
		    
		    object_1.localBest=new double[object_1.population_size];
		    object_1.localBestParticles=new int[object_1.population_size][in_train_major.numInstances()];
		    object_1.globalBest=Double.MIN_VALUE;
		    object_1.globalBestParticles=new int[in_train_major.numInstances()];
		    object_1.velocity=new double[object_1.population_size][in_train_major.numInstances()];
		    object_1.particles=new int[object_1.population_size][in_train_major.numInstances()];
		    System.out.println("---------");
		    System.out.println("Fold = "+fold);
		    
		    //initialization with the prospective of majority class
		    
		    //initalize the particle position
		    int dimension=in_train_major.numInstances();
		    for(int x=0;x<object_1.population_size;x++)
		    {
		    	for(int y=0;y<dimension;y++)
		    	{
		    		if(object_1.random.nextDouble()>0.5)
		    		{
		    			object_1.particles[x][y]=1;
		    		}
		    		else
		    		{
		    			object_1.particles[x][y]=0;
		    		}
		    	}
		    }
		    
		    for(int count=0;count<object_1.population_size;count++)
	    	{
	           object_1.localBest[count]=Double.MIN_VALUE;  		
	    	}
	    	object_1.globalBest=Double.MIN_VALUE;
		    
	    	
	    	//find the optimum solutions
	    	
	    	for(int iterator=0;iterator<object_1.iteration_number;iterator++)
	    	{
	    		for(int x=0;x<object_1.population_size;x++)
	    		{
	    			double testAUC=0.0d;
	    			Instances optimizedSet=new Instances(object_1.training_set);
	    			optimizedSet.delete();
	    			
	    			for(int i=0;i<in_train_minor.numInstances();i++)
	    				optimizedSet.add(in_train_minor.instance(i));
	    			
	    			for(int i=0;i<in_train_major.numInstances();i++)
	    				if(object_1.particles[x][i]==1)
	    				{
	    					optimizedSet.add(in_train_major.instance(i));
	    				}
	    			
	    			J48 c=new J48();
	    			try {
						c.buildClassifier(optimizedSet);
					    Evaluation evaluation=new Evaluation(optimizedSet);
					    evaluation.evaluateModel(c, object_1.In_testig_set);
					    testAUC=evaluation.areaUnderROC(1);
	    			} catch (Exception e) {
						e.printStackTrace();
					}
	    			
	    			if(object_1.localBest[x]<testAUC)
	    			{
	    				for(int y=0;y<dimension;y++)
	    				{
	    					object_1.localBestParticles[x][y]=object_1.particles[x][y];
	    				}
	    				object_1.localBest[x]=testAUC;
	    			}
	    			
	    			if(object_1.globalBest<testAUC)
	    			{
	    				for(int y=0;y<dimension;y++)
	    				{
	    					object_1.globalBestParticles[y]=object_1.particles[x][y];
	    				}
	    				object_1.globalBest=testAUC;
	    			}
	    		}
	    		
	    		for(int x=0;x<object_1.population_size;x++)
	    		{
	    			for(int y=0;y<dimension;y++)
	    			{
	    				double r1=object_1.random.nextDouble();
	    				double r2=object_1.random.nextDouble();
	    				
	    				object_1.velocity[x][y]=object_1.w*object_1.velocity[x][y]+object_1.c1*r1*(object_1.localBestParticles[x][y]-object_1.particles[x][y])+object_1.c2*r2*(object_1.globalBestParticles[y]-object_1.particles[x][y]);
	    				
	    				if(object_1.velocity[x][y]>object_1.max_velocity)
	    					object_1.velocity[x][y]=object_1.max_velocity;
	    				
	    				if(object_1.velocity[x][y]<object_1.min_velocity)
	    					object_1.velocity[x][y]=object_1.min_velocity;
	    				
	    				if(object_1.random.nextDouble()>=1/(1+Math.exp(-object_1.velocity[x][y])))
	    				      object_1.particles[x][y]=0;
	    				else
	    					object_1.particles[x][y]=1;
	    
	    			}
	    		}
	    	}
	    	
	    	//store optimized solutions
	    	for(int i=0;i<object_1.population_size;i++)
	    	{
	    		Instances optimizedSet=new Instances(object_1.training_set);
	    		optimizedSet.delete();
	    		//Instances sso_final_train_set=
	    		for(int i1=0;i1<in_train_minor.numInstances();i1++)
	    			optimizedSet.add(in_train_minor.instance(i1));
	    		
	    		for(int i1=0;i1<in_train_major.numInstances();i1++)
	    			if(object_1.localBestParticles[i][i1]==1)
	    				optimizedSet.add(in_train_major.instance(i1));
                
	    		if(fold==0)
	    			object_1.optimum_sets.add(optimizedSet);
	    		else
	    		{
	    			for(int j=0;j<optimizedSet.numInstances();j++)
	    			{
	    				object_1.optimum_sets.get(i).add(optimizedSet.instance(j));
	    			}
	    		}
	    		
	    	}	
	    	System.out.println("\n\nReadings with SSO-PSO on majority class : ");
	    	for(int i=0;i<object_1.localBest.length;i++)
	    	{
	    		object_1.avgFitness+=object_1.localBest[i];
	    	}
	    	System.out.println("AUC : "+object_1.avgFitness/object_1.localBest.length);
	    	object_1.avgFitness=0;
		}
		
		

    	object_1.hash_table=new Hashtable<String,Integer>();
    	
    	for(int i=0;i<object_1.optimum_sets.size();i++)
    	{
    		Instances set=object_1.optimum_sets.get(i);
    		
    		for(int j=0;j<set.numInstances();j++)
    		{
    			if(set.instance(j).classValue()==0)
    			{
    				String inst=set.instance(j).toString();
    				
    				if(object_1.hash_table.containsKey(inst))
    				{
    					Integer C=(Integer)object_1.hash_table.get(inst);
    					int c=C.intValue();
    					c++;
    					
    					if(object_1.max_count<c)
    						object_1.max_count=c;
    					
    					C=new Integer(c);
    					object_1.hash_table.put(inst, C);
    				}
    				else
    				{
    					object_1.hash_table.put(inst, 1);
    				}
    			}
    		}
    	}
    	
    	String balanced_file="balanceTrain_"+file_name;
    	
		//Store the more frequent data samples
		
		int max=0,min=0,iterator=0;
		List majority_VIP=new LinkedList();
		Iterator itr=object_1.hash_table.keySet().iterator();
		while(itr.hasNext())
		{
			String inst=(String) itr.next().toString();
			String value=object_1.hash_table.get(inst).toString();
			Integer val=new Integer(value);
			majority_VIP.add(new Object[] {val.intValue(),inst});
		}
		
		System.out.println("\n\n------------------------------------------------------------------------------------");
		
		int num_of_minority_instances=0;
		
		for(int i=0;i<object_1.training_set.numInstances();i++)
			if(object_1.training_set.instance(i).classValue()==1)
				num_of_minority_instances++;
		
		itr=majority_VIP.iterator();
		max=(Integer)((Object[])itr.next())[0];
		Instance majority_class_data_sample=null;
		itr=null;
		itr=majority_VIP.iterator();
		Instances majority_class_instances=new Instances(object_1.training_set);
		majority_class_instances.delete();
		
		while(itr.hasNext() && iterator<(num_of_minority_instances))
		{
			Object[] data=((Object[])itr.next());
			String data_sample=(String)data[1];
			String []values=data_sample.split(",");
			double values_attribute_for_a_sample[]=new double[values.length];
			for(int i=0;i<values.length;i++)
			  {
				  if(i!=object_1.training_set.classIndex())
				  {
				       values_attribute_for_a_sample[i]=Double.parseDouble(values[i]);
				  }
				  else if(i==object_1.training_set.classIndex())
				  {
					  int class1=(int)Double.parseDouble(values[i]);
					  values_attribute_for_a_sample[i]=class1;
				  }
			  }
			majority_class_data_sample=new Instance(1.0, values_attribute_for_a_sample);
			majority_class_instances.add(majority_class_data_sample);
			iterator++;
		}

		Instances minority_class=new Instances(object_1.training_set);
		minority_class.delete();
		for(int i=0;i<object_1.training_set.numInstances();i++)
		  {
			  if(object_1.training_set.instance(i).classValue()==1)
			  {
				  Instance instance=object_1.training_set.instance(i);
				  minority_class.add(instance);
			  }
		  }
		object_1.training_set.delete(); 
		
		  for(int i=0;i<majority_class_instances.numInstances();i++)
		  {
			  object_1.training_set.add(majority_class_instances.instance(i));
		  }
		  for(int i=0;i<minority_class.numInstances();i++)
		  {
			  object_1.training_set.add(minority_class.instance(i));
		  }
		  
		  copy_training_set = new Instances(object_1.training_set);
		  copy_training_set.stratify(2);
		  object_1.optimum_sets=new ArrayList<Instances>();
		  object_1.In_training_set=null;
		  object_1.In_testig_set=null;
		  for (int fold = 0; fold < 2; fold++) {
				
			    object_1.In_training_set=copy_training_set.trainCV(2, fold);
				Instances test=copy_training_set.testCV(2, fold);
				object_1.In_testig_set=new Instances(test);
				
				Instances in_train_minor=new Instances(object_1.In_training_set);
				in_train_minor.delete();
				Instances in_train_major=new Instances(object_1.In_training_set);
			    in_train_major.delete();
			    			    
			    for(int i=0;i<object_1.In_training_set.numInstances();i++)
			    {
			    	Instance instance=object_1.In_training_set.instance(i);
			    	if(instance.classValue()==1)
			    	{
			    		in_train_minor.add(instance);
			    	}
			    	else
			    	{
			    		in_train_major.add(instance);
			    	}
			    }
			    
			    Instances minority_synthetic_combo=object_1.SMOTE();   //oversample the minority class
				System.out.println("\nNumber of sythetic data samples generated : "+minority_synthetic_combo.numInstances());
				for(int i=0;i<minority_synthetic_combo.numInstances();i++)
				  {
					in_train_minor.add(minority_synthetic_combo.instance(i));
					object_1.training_set.add(minority_synthetic_combo.instance(i));
				  }
			    
			    for(int i=0;i<test.numInstances();i++)
			    {
			    	Instance instance=test.instance(i);
			    	if(instance.classValue()==1)
			    	{
			    		object_1.In_testig_set.add(instance);
			    		object_1.In_testig_set.add(instance);
			    	}
			    }
			    
			    object_1.localBest=new double[object_1.population_size];
			    object_1.localBestParticles=new int[object_1.population_size][in_train_minor.numInstances()];
			    object_1.globalBest=Double.MIN_VALUE;
			    object_1.globalBestParticles=new int[in_train_minor.numInstances()];
			    object_1.velocity=new double[object_1.population_size][in_train_minor.numInstances()];
			    object_1.particles=new int[object_1.population_size][in_train_minor.numInstances()];
			    System.out.println("---------");
			    System.out.println("Fold = "+fold);
			    
			    //initialization with the prospective of minority class
			    
			    //initalize the particle position
			    int dimension=in_train_minor.numInstances();
			    for(int x=0;x<object_1.population_size;x++)
			    {
			    	for(int y=0;y<dimension;y++)
			    	{
			    		if(object_1.random.nextDouble()>0.5)
			    		{
			    			object_1.particles[x][y]=1;
			    		}
			    		else
			    		{
			    			object_1.particles[x][y]=0;
			    		}
			    	}
			    }
			    
			    for(int count=0;count<object_1.population_size;count++)
		    	{
		           object_1.localBest[count]=Double.MIN_VALUE;  		
		    	}
		    	object_1.globalBest=Double.MIN_VALUE;
			    
		    	
		    	//find the optimum solution
		    	
		    	for( iterator=0;iterator<object_1.iteration_number;iterator++)
		    	{
		    		for(int x=0;x<object_1.population_size;x++)
		    		{
		    			double testAUC=0.0d;
		    			Instances optimizedSet=new Instances(object_1.training_set);
		    			optimizedSet.delete();
		    			
		    			//copy positive samples
		    			for(int i=0;i<in_train_major.numInstances();i++)
		    				optimizedSet.add(in_train_major.instance(i));
		    			
		    			for(int i=0;i<in_train_minor.numInstances();i++)
		    				if(object_1.particles[x][i]==1)
		    				{
		    					optimizedSet.add(in_train_minor.instance(i));
		    				}
		    			
		    			J48 c=new J48();
		    			try {
							c.buildClassifier(optimizedSet);
						    Evaluation evaluation=new Evaluation(optimizedSet);
						    evaluation.evaluateModel(c, object_1.In_testig_set);
						    testAUC=evaluation.areaUnderROC(1);
		    			} catch (Exception e) {
							e.printStackTrace();
						}
		    			
		    			if(object_1.localBest[x]<testAUC)
		    			{
		    				for(int y=0;y<dimension;y++)
		    				{
		    					object_1.localBestParticles[x][y]=object_1.particles[x][y];
		    				}
		    				object_1.localBest[x]=testAUC;
		    			}
		    			
		    			if(object_1.globalBest<testAUC)
		    			{
		    				for(int y=0;y<dimension;y++)
		    				{
		    					object_1.globalBestParticles[y]=object_1.particles[x][y];
		    				}
		    				object_1.globalBest=testAUC;
		    			}
		    		}
		    		
		    		for(int x=0;x<object_1.population_size;x++)
		    		{
		    			for(int y=0;y<dimension;y++)
		    			{
		    				double r1=object_1.random.nextDouble();
		    				double r2=object_1.random.nextDouble();
		    				
		    				object_1.velocity[x][y]=object_1.w*object_1.velocity[x][y]+object_1.c1*r1*(object_1.localBestParticles[x][y]-object_1.particles[x][y])+object_1.c2*r2*(object_1.globalBestParticles[y]-object_1.particles[x][y]);
		    				
		    				if(object_1.velocity[x][y]>object_1.max_velocity)
		    					object_1.velocity[x][y]=object_1.max_velocity;
		    				
		    				if(object_1.velocity[x][y]<object_1.min_velocity)
		    					object_1.velocity[x][y]=object_1.min_velocity;
		    				
		    				if(object_1.random.nextDouble()>=1/(1+Math.exp(-object_1.velocity[x][y])))
		    				      object_1.particles[x][y]=0;
		    				else
		    					object_1.particles[x][y]=1;
		    
		    			}
		    		}
		    	}
		    	
		    	//store optimized solutions
		    	for(int i=0;i<object_1.population_size;i++)
		    	{
		    		Instances optimizedSet=new Instances(object_1.training_set);
		    		optimizedSet.delete();
		    		for(int i1=0;i1<in_train_major.numInstances();i1++)
		    			optimizedSet.add(in_train_major.instance(i1));
		    		
		    		for(int i1=0;i1<in_train_minor.numInstances();i1++)
		    			if(object_1.localBestParticles[i][i1]==1)
		    				optimizedSet.add(in_train_minor.instance(i1));
	                
		    		if(fold==0)
		    			object_1.optimum_sets.add(optimizedSet);
		    		else
		    		{
		    			for(int j=0;j<optimizedSet.numInstances();j++)
		    			{
		    				object_1.optimum_sets.get(i).add(optimizedSet.instance(j));
		    			}
		    		}
		    		
		    	}	
		        
		    	System.out.println("\n\nReadings with SMOTE and SSO-PSO on minority class : ");
		    	for(int i=0;i<object_1.localBest.length;i++)
		    	{
		    		object_1.avgFitness+=object_1.localBest[i];
		    	}
		    	System.out.println("AUC : "+object_1.avgFitness/object_1.localBest.length);
		    	object_1.avgFitness=0;
			}
			
	    	object_1.balance_set=new Instances(object_1.training_set);
	    	object_1.balance_set.delete();
	    	
	    	 int majorsize=0;
	    	for(int i=0;i<object_1.training_set.numInstances();i++)
	    	{
	    		if(object_1.training_set.instance(i).classValue()==0)
	    		{
	    			object_1.balance_set.add(object_1.training_set.instance(i));
	    			majorsize++;
	    		}
	    	}
	    	
	    	object_1.hash_table=new Hashtable<String,Integer>();
	    	
	    	for(int i=0;i<object_1.optimum_sets.size();i++)
	    	{
	    		Instances set=object_1.optimum_sets.get(i);
	    		
	    		for(int j=0;j<set.numInstances();j++)
	    		{
	    			if(set.instance(j).classValue()==1)
	    			{
	    				String inst=set.instance(j).toString();
	    				
	    				if(object_1.hash_table.containsKey(inst))
	    				{
	    					Integer C=(Integer)object_1.hash_table.get(inst);
	    					int c=C.intValue();
	    					c++;
	    					
	    					if(object_1.max_count<c)
	    						object_1.max_count=c;
	    					
	    					C=new Integer(c);
	    					object_1.hash_table.put(inst, C);
	    				}
	    				else
	    				{
	    					object_1.hash_table.put(inst, 1);
	    				}
	    			}
	    		}
	    	}
			
			
			//stroe the balanced datasets
			BufferedWriter bw;
			try {
				bw = new BufferedWriter(new FileWriter(directory_name+balanced_file));
				bw.write(object_1.balance_set.toString());
			    bw.newLine();
			
	    	
			int count=0;
			Iterator<String> itr1;
			itr1=object_1.hash_table.keySet().iterator();
			while(itr1.hasNext())
			{
				String inst=(String)itr1.next().toString();
				String value=object_1.hash_table.get(inst).toString();
				
				if(count<majorsize)
				{
					
					bw.write(inst);
					bw.newLine();
					count++;
				}
				
			}
			bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		  }

	private Instances SMOTE() {
		Instances inTrainMinor;
 		Instances minority_examples;
 		Instances minority_examples_synthetic=new Instances(this.In_training_set);
 		minority_examples_synthetic.delete();
 		
 		inTrainMinor=new Instances(this.In_training_set);
 		inTrainMinor.delete();
 		
 		minority_examples=new Instances(this.In_training_set);
 		minority_examples.delete();
 		
 		int k_neighbors=9;
 		
 		List distanceToInstance=null; 
 		int minIndex=1;  // minority class label
 		
 		 
 		 
 		float percentageRemainder=0.25f;     //it determines the percentage of synthetic data samples to be generated. Here, 25% means 0.25 and 85% means 0.85
 	    double distance=0.0d;
 		for(int i=0;i<this.In_training_set.numInstances();i++)
 		{
 			if(this.In_training_set.instance(i).classValue()==minIndex)
 			{
 				minority_examples.add(this.In_training_set.instance(i));
 			}
 		}
 		
 		System.out.println("\n\nNumber of minority examples in training set "+minority_examples.numInstances());
 		int extraIndicesCount=(int) Math.floor(percentageRemainder*minority_examples.numInstances());  // determining the number of synthetic minority class data sample to be generated
 		System.out.println("Number of synthetic samples to be include :"+extraIndicesCount);
 		Instances mArray[]=new Instances[k_neighbors];     // Considering 9 neighbors for minority class data sample
        int number_of_attributes=this.In_training_set.numAttributes();
        
        Instance nnArray[]=new Instance[9];
        System.out.println("Number of attributes "+number_of_attributes);
        
 		for(int i=0;i<minority_examples.numInstances();i++)
 		{
 			distanceToInstance =new LinkedList(); 
 			Instance instanceI=minority_examples.instance(i);
 			for(int j=0;j<minority_examples.numInstances();j++)
 			{
 				Instance instanceJ=minority_examples.instance(j);
 				
 				if((instanceI.toString()).equals((instanceJ.toString()))==false)  
 				{   					
 					 for(int attribute_number=0;attribute_number<number_of_attributes-1;attribute_number++)
 					 {
 						distance+= Math.pow(instanceI.value(attribute_number)-instanceJ.value(attribute_number),2);
 				     }
 					distance= Math.pow(distance, 0.5);
 					distanceToInstance.add(new Object[] {distance, instanceJ});
 					distance=0.0d; 
 				}
 			}
 			
 		   // sort the neighbors according to distance
 			  Collections.sort(distanceToInstance, new Comparator() {
 					public int compare(Object o1, Object o2) {
 					  double distance1 = (Double) ((Object[]) o1)[0];
 					  double distance2 = (Double) ((Object[]) o2)[0];
 				          return Double.compare(distance1, distance2);
 					}
 				      });

 			  Iterator entryIterator = distanceToInstance.iterator();
 			  int j1=0;
 			  while(entryIterator.hasNext() && j1<k_neighbors)
 			  {	
				nnArray[j1] = (Instance) ((Object[])entryIterator.next())[1];
 				j1++;
 			  }
 			  int nn=this.random.nextInt(k_neighbors);   // K=9 nearest neighbors i.e, random value between 0 to k
 			  double[] values=new double[this.In_training_set.numAttributes()];
 			  for(int attribute_number=0;attribute_number<this.In_training_set.numAttributes();attribute_number++)
 			  {
 				  if(this.In_training_set.classIndex()!=attribute_number)
 				  {
 				  double dif=nnArray[nn].value(attribute_number)-instanceI.value(attribute_number);
 				  double gap=this.random.nextDouble();  // random value
 				  values[attribute_number]=instanceI.value(attribute_number)+dif*gap;
 				  }
 				  
 				  if(this.In_training_set.classIndex()==attribute_number)
 				  {
 					  values[attribute_number]=minIndex;
 				  }
 			  }
 			  
 			 Instance synthetic = new Instance(1.0, values);
 			 if(minority_examples_synthetic.numInstances()<extraIndicesCount)
 			          minority_examples_synthetic.add(synthetic);
 			distanceToInstance=null;
     
 		}
		return minority_examples_synthetic;
	}
		  
		  


}

