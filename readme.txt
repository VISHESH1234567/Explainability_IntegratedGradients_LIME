Task 0
    1) Getting datasets and other setup files
        a. Download the IAPRTC dataset (https://drive.google.com/file/d/1wiuOT3bG6GocnYn5BCwjKT2isWaSZ2el/view?usp=drive_link), provided by Dr. Yashaswi Verma, IIT Jodhpur.
        b. Download the raw iarptc12 dataset from https://www.imageclef.org/photodata
	c. Pretrained Word Vector: https://nlp.stanford.edu/projects/glove/
	d. Extra Setup Files from(unzip): https://drive.google.com/file/d/1kWqPizSZ_t7AOAdGysWJam5Fi8eEFOK7/view?usp=sharing
    2) Generating train and test files - Run either of the setup_GloveEmbeddings.ipynb or setup_SbertEmbeddings.ipynb  depending on the pipeline to generate train.svm and test.svm
    3) Cloning AnnexML repository - In the AnnexML directory, write command "git clone https://github.com/yahoojapan/AnnexML"
    4) Directory Structure
       -- 📁 IAPRTC
       -- 📁 iaprtc12
       -- 📁 AnnexML
       -- 📁 glove.6B
       -- 📄setup_SbertEmbeddings.ipynb
       -- 📄setup_GloveEmbeddings.ipynb
       -- 📄train.svm
       -- 📄test.svm
       -- 📄IntegratedGradients.ipynb
       -- 📄LIME.ipynb
    5) Build
	a. If using Windows, do the following changes:
	    -> Replace "size_t num_cluster = std::max(data_vec.size() / 6000, 1LU);" with "size_t num_cluster = std::max(data_vec.size() / 6000, static_cast<size_t>(1));" in src/AnnexML.cc
	    -> Replace "for (size_t i = 0; i < std::min(10LU, idx_len_vec.size()); ++i) { fprintf(stderr, "%lu, ", idx_len_vec[i]); }" with "for (size_t i = 0; i 	     	            		< std::min(static_cast<size_t>(10), idx_len_vec.size()); ++i) { fprintf(stderr, "%lu, ", idx_len_vec[i]); }" in src/DataPartitioner.cc
	    -> Add "#ifdef _WIN32
                    #include <malloc.h>
                    #define posix_memalign(ptr, align, size) (((*(ptr)) = _aligned_malloc(size, align)) ? 0 : errno)
                    #else
                    #include <stdlib.h>
                    #endif" at top of src/NGT.cc and src/LLEmbedding.cc
            -> Add "CXXFLAGS += -DWIN32 -D_USE_MATH_DEFINES" in src/Makefile
	b. If your CPUs do not support FMA instruction set, you should comment out the line "CXXFLAG += -DUSEFMA -mfma" in src/Makefile before making.
	c. Run the command "make -C src/ annexml" on terminal
    6) AnnexML Train and Predict
	a. Data Format - AnnexML takes multi-label svmlight / libsvm format.
	   E.g. - "
	           32,50,87 1:1.9 23:0.48 79:0.63
		   50,51,126 4:0.71 23:0.99 1005:0.08 1018:2.15
		  "
	b. Training:
	    -> Model parameters and some file paths are specified in a JSON file - "AnnexML-example.json" or command line arguments.
            -> Run one of the following commands "
						  src/annexml train annexml-example.json
 						  src/annexml train annexml-example.json num_thread=32   # use 32 CPU threads for training
						  src/annexml train annexml-example.json cls_type=0   # use k-means clustering for data partitioning
						 "
	C. Predictions: Run one of the following commands "
							   src/annexml predict annexml-example.json
							   src/annexml predict annexml-example.json num_learner=4 num_thread=1   # use only 4 learners and 1 CPU thread for prediction
							   src/annexml predict annexml-example.json pred_type=0   # use brute-force cosine calculation
							  "
    7) AnnexML Model Parameters and File paths:
    emb_size          Dimension size of embedding vectors
	num_learner       Number of learners (or models) for emsemble learning
	num_nn            Number of (approximate) nearest neighbors used in training and prediction
	cls_type          Algorithm type used for data partitioning
	                  1 : learning procedure which finds min-cut of approximate KNNG
	                  0 : k-means clustering
	cls_iter          Number of epochs for data partitioning algorithms
	emb_iter          Number of epochs for learning embeddings
	label_normalize   Label vectors are normalized or not
	eta0              Initial value of AdaGrad learning rate adjustement
	lambda            L1-regularization parameter of data partitioning (only used if cls_type = 1)
	gamma             Scaling parameter for cosine ([-1, 1] to [-gamma, gamma]) in learning embeddings
	pred_type         Algorithm type used for prediction of k-nearest neighbor classifier
	                  1 : approximate nearest neighbor search method which explores learned KNNG
	                  0 : brute-force calculation
	num_edge          Number of direct edges per vertex in learned KNNG (only used if pred_type = 1)
	search_eps        Parameter for exploration of KNNG (only used if pred_type = 1)
	num_thread        Number of CPU threads used in training and prediction
	seed              Random seed
	verbose           Vervosity level (ignore if num_thread > 1)
	
	train_file        File path of training data
	predict_file      File path of prediction data
	model_file        File path of output model
	result_file       File path of prediction result

    8) Dependencies: AnnexML includes the software - (2-clause BSD license) picojson(https://github.com/kazuho/picojson)

__________________________________________________________________________________________________________________________________________________________________________________________
Task 1
    1) Dependencies/Libraries:
	numpy
	pandas
	subprocess
	os
	matplotlib
	scipy
	re
	chardet
	torch
	sentence_transformers
    transformers
	nltk
	torch
	Ipython
    2) Run either setup_GloveEmbeddings.ipynb or setup_SbertEmbeddings.ipynb file to get train.svm and text.svm files.
       Train the AnnexML model by running the following command inside the assignment directory: "AnnexML/src/annexml train AnnexML/annexml-example.json train_file=train.svm model_file=annexml-model-example.bin predict_file=test.svm num_threads=31"
    3) Parameters in IntegratedGradients.ipynb file:
	pipeline	1: To explain the AnnexML model using SBERT Sentence Embeddings Integrated Gradients
				2: To explain the AnnexML model using Glove Word Embeddings Integrated Gradients
	lines		 list containing indicies for lines from IAPRTC/iapr_test_list.txt
	test_path       Location to store the SVM format of the test data to be explained
	test_text_path  Location to store the English text of the test data to be explained
	num_steps		Factor to scale the test data to calculate gradients at various scales
	step_size		Amount of small perturbations in scaled data to approximate gradients
	scaled_test_path	Location to store the scaled values of test data to be explained
	test_pred_path		Location to store the predictions made by the AnnexML model on the test data
	scaled_pred_path	Location to store the predictions made by the AnnexML model on the scaled test data
    4) Run the IntegratedGradients.ipynb file
_________________________________________________________________________________________________________________________________________________________________________________________
Task 2
	1) Dependencies/Libraries:
	numpy
	pandas
	subprocess
	os
	matplotlib
	scipy
	re
	chardet
	sentence_transformers
	Ipython
    2) Run setup_SbertEmbeddings.ipynb file to get train.svm and text.svm files. 
       Train the AnnexML model by running the following command inside the assignment directory: "AnnexML/src/annexml train AnnexML/annexml-example.json train_file=train.svm model_file=AnnexML/annexml-model-example.bin predict_file=test.svm num_threads=31"
    3) Parameters in IntegratedGradients.ipynb file:
	explainrows			Number of rows of test.svm to be explained using LIME method
	samples_per_row			Number of perturbed samples to generate per test datapoint
	pick_probab			Chance of picking a word when generating perturbed samples
	test_path			Location to store the SVM format of the test data to be explained
	test_text_path			Location to store the English text of the test data to be explained
	perturbed_test_path		Location to store the perturbed samples of test data to be explained
	perturbed_interpret_path	Location to store the interpretable form of the perturbed samples of the test data to be explained
	test_pred_path			Location to store the predictions made by the AnnexML model on the test data to be explained
	perturbed_pred_path		Location to store the predictions made by the AnnexML model on the perturbed samples
    4) Run the LIME.ipynb file
