#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <Eigen/Dense>
#include <time.h>
using namespace Eigen;
using namespace std;

unsigned int feature_size;
unsigned int feature_pca_size;
unsigned int n_set;
unsigned int n_test;
unsigned int one_persion_pick;
unsigned int one_persion_least;
MatrixXi test_record; //MatrixXi::(, )
void jointbayesian_get_train_test_data(const char* labels_name, const char* feature_name, const char* labels_test_name, const char* feature_test_name, const char* labels_train_name, const char* feature_train_name);
void jointbayesain_PCA(const char* labels_name, const char* feature_name, const char* feature_pca_name, const char* feature_mean_name, const char* pca_matrix_name);
void jointbayesain_AG(const char* labels_name, const char* feature_name, const char* A_name, const char* G_name);
void jointbayesain_ratio(const char* labels_test_name, const char* feature_test_name, const char* feature_mean_name, const char* pca_matrix_name, const char* A_name, const char* G_name, const char* postivepair_ratio_name, const char* nagtivepair_ratio_name);
void jointbayesain_ratio_old(const char* labels_test_name, const char* feature_test_name, const char* feature_mean_name, const char* pca_matrix_name, const char* A_name, const char* G_name, const char* ratio_name, const char* ROC_name);
int main()
{
	cout << "how many feature_size :"<<endl;
	cin >> feature_size;
	cout << "how many feature_pca_size :" << endl;
	cin >> feature_pca_size;
	cout << "how many max images of each class,(default min image=4) :" << endl;
	cin >> n_set;
	cout << "how many class of dataset :" << endl;
	cin >> n_test;
	cout << "how many images of each class about test step (default=2) :" << endl;
	cin >> one_persion_pick;
	cout << "how many images of each class about train step, at lease 2 images(default=2) :" << endl;
	cin >> one_persion_least;


	jointbayesian_get_train_test_data("label.txt", "feature.txt", "label_test.txt", "feature_test.txt", "label_train.txt", "feature_train.txt");
	jointbayesain_PCA("label_train.txt", "feature_train.txt", "feature_pca.txt", "feature_mean.txt", "eigenvector_need.txt");
	jointbayesain_AG("label_train.txt", "feature_pca.txt", "A.txt", "G.txt");
	jointbayesain_ratio("label_test.txt", "feature_test.txt", "feature_mean.txt", "eigenvector_need.txt", "A.txt", "G.txt", "ratio_p.txt","ratio_n");

	system("pause");
}
void jointbayesian_get_train_test_data(const char* labels_name, const char* feature_name, const char* labels_test_name, const char* feature_test_name, const char* labels_train_name, const char* feature_train_name)
{
	cout << "train test data start!" << endl;
	vector < int > labels;
	bool check = 1;
	string labels_name_r = labels_name;
	ifstream finlabels(labels_name_r);
	int lebals_data_r;
	if (!finlabels) {
		cout << "can not read labels" << endl;
		check = 0;
	}
	else
	{
		cout << "read labels" << endl;
		while (!finlabels.eof())
		{
			finlabels >> lebals_data_r;
			labels.push_back(lebals_data_r);
			finlabels.get();
			if (finlabels.peek() == '\n')
			{
				break;
			}
		}
	}
	finlabels.close();
	cout << "total class :" << labels.size() << endl;
	vector<vector<float>> X(labels.size(), vector<float>(feature_size));
	string feature_name_r = feature_name;
	ifstream finfeature(feature_name_r);
	if (!finfeature) {
		cout << "can not read feature" << endl;
		check = 0;
	}
	else
	{
		cout << "read feature" << endl;
		for (unsigned int j = 0; j < labels.size(); j++)
		{
			for (unsigned int i = 0; i < feature_size; i++)
			{
				finfeature >> X[j][i];
			}
		}
	}
	finfeature.close();
	//
	if (check == 1)
	{
		vector < int > labels_class(labels);
		sort(labels_class.begin(), labels_class.end());
		vector<int>::iterator iter = unique(labels_class.begin(), labels_class.end());
		labels_class.erase(iter, labels_class.end());
		vector<vector<int>> labels_location(labels_class.size(), vector<int>(n_set + 1));
		unsigned int more_than_set = 0;
		unsigned int more_than_least = 0;
		for (unsigned int i = 0; i < labels_class.size(); i++)
		{
			unsigned int k = 0;
			for (unsigned int j = 0; j < labels.size(); j++)
			{
				if (labels_class[i] == labels[j])
				{
					if (k < n_set)
					{
						labels_location[i][k] = j + 1;//labelsŠìžm¬ö¿ý,žò¹s°Ï§O,¹ê»Ú­n¥Î®ÉŽî1
						k = k + 1;
					}
				}
			}
			labels_location[i][n_set] = k;//¬ö¿ýšC€HŠ³ŽX±i
			if (k >= one_persion_least)
			{
				more_than_least = more_than_least + 1;
				if (k  >= (one_persion_pick + one_persion_least))
				{
					more_than_set = more_than_set + 1;
				}
			}
		}
		cout << "classification over" << endl;
		//cout << "more_than_set" << more_than_set << endl;
		//cout << "ŠP€@€H©âš«one_persion_pick±iŒÆ«á³ÑŸl±iŒÆ€j©óone_persion_least€§classŒÆ" << more_than_set << endl;
		if (n_test>more_than_set )
		{
			n_test = more_than_set;
			cout << "­original test pair is too much, change to :" << n_test << endl;
		}

		
		//vector<vector<int>> labels_test2(n_test, vector<int>(one_persion_pick + 1));//label_number, location,location
		//vector<vector<int>> labels_train2(more_than_least, vector<int>(n_set + 2));//label_number, location,location....location,how many
		vector<vector<int>> labels_test(n_test, vector<int>(one_persion_pick + 1));//label_number, location,location
		vector<vector<int>> labels_train(more_than_least, vector<int>(n_set + 2));//label_number, location,location....location,how many
		vector<int> labels_test_temp(one_persion_pick);
		unsigned int i_labels_test = 0;
		unsigned int i_labels_train = 0;
		int num1=17;
		int num2 = 19;
		cout << "total class" << labels_class.size() << endl;
		for (unsigned int i = 0; i < labels_class.size(); i++)
		{
			if (labels_location[i][n_set] >= (one_persion_pick + one_persion_least) && i_labels_test<n_test)
			{
				labels_test[i_labels_test][0] = labels_class[i];
				for (unsigned int t = 1; t < (one_persion_pick + 1); t++)
				{
					//
					unsigned int j = t * num1;
					if (labels_location[i][n_set] >= num1)
					{
						if (labels_location[i][n_set] % num1 == 0)
						{
							j = t * num2;
						}
					}
					else
					{
						if (20%labels_location[i][n_set] == 0)
						{
							j = t * num2;
						}
					}
					if (j>labels_location[i][n_set])
					{
						j = j%labels_location[i][n_set];
					}
					labels_test[i_labels_test][t] = labels_location[i][j - 1];
					labels_test_temp[t - 1] = labels_location[i][j-1];
					//
					//labels_test2[i_labels_test][t] = labels_location[i][t - 1];
				}
				labels_train[i_labels_train][0] = labels_class[i];
				//labels_train2[i_labels_train][0] = labels_class[i];
				unsigned int m = 1;
				unsigned int m2 = 1;
				/*
				for (unsigned int j = one_persion_pick; j < n_set; j++)
				{
					labels_train2[i_labels_train][m2] = labels_location[i][j];
					m2 = m2 + 1;
				}
				*/
				for (unsigned int j = 1; j <= labels_location[i][n_set]; j++)
				{
					vector<int>::iterator it;
					it = find(labels_test_temp.begin(), labels_test_temp.end(), j);
					if (it != labels_test_temp.end()){
						
					}
					else{
						labels_train[i_labels_train][m] = labels_location[i][j-1];
						m = m + 1;
					}
				}
				labels_train[i_labels_train][n_set + 1] = labels_location[i][n_set] - one_persion_pick;
				//labels_train2[i_labels_train][n_set + 1] = labels_location[i][n_set] - one_persion_pick;
				i_labels_test = i_labels_test + 1;
				i_labels_train = i_labels_train + 1;
			}
			else if (labels_location[i][n_set] >= (one_persion_pick + one_persion_least) && i_labels_test >= n_test)
			{
				labels_train[i_labels_train][0] = labels_class[i];
				for (unsigned int j = 0; j < n_set; j++)
				{
					labels_train[i_labels_train][j + 1] = labels_location[i][j];
				}
				i_labels_train = i_labels_train + 1;
			}

			else if (labels_location[i][n_set] < (one_persion_pick + one_persion_least) && labels_location[i][n_set] >= one_persion_least)
			{
				labels_train[i_labels_train][0] = labels_class[i];
				for (unsigned int j = 0; j < n_set; j++)
				{
					labels_train[i_labels_train][j + 1] = labels_location[i][j];
				}
				labels_train[i_labels_train][n_set + 1] = labels_location[i][n_set];
				i_labels_train = i_labels_train + 1;
			}
		}
		test_record = MatrixXi::Zero(n_test, one_persion_pick + 1);
		for (int i = 0; i <n_test; i++)
		{
			for (int j = 0; j < one_persion_pick + 1; j++)
			{
				test_record(i, j) = labels_test[i][j];
			}
		}


		string labels_train_name_r = labels_train_name;
		fstream finlabels_train;
		finlabels_train.open(labels_train_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finlabels_train)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "can not read: " << labels_train_name_r << endl;
		}
		else
		{
			cout << "read: " << labels_train_name_r << endl;
			for (int i = 0; i < more_than_least; i++)
			{
				for (int j = 0; j < labels_train[i][n_set + 1]; j++)
				{
					finlabels_train << labels_train[i][0];
					finlabels_train << endl;
				}
			}
		}
		finlabels_train.close();
		string labels_test_name_r = labels_test_name;
		fstream finlabels_test;
		finlabels_test.open(labels_test_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finlabels_test)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "can not read:" << labels_test_name_r << endl;
		}
		else
		{
			cout << "read:" << labels_test_name_r << endl;
			for (int i = 0; i < n_test; i++)
			{
				for (int j = 0; j < one_persion_pick; j++)
				{
					finlabels_test << labels_test[i][0];
					finlabels_test << endl;
				}
			}
		}
		finlabels_test.close();
		
		
		string feature_train_name_r = feature_train_name;
		fstream finfeature_train;
		finfeature_train.open(feature_train_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finfeature_train)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "can not read:" << feature_train_name_r << endl;
		}
		else
		{
			cout << "read:" << feature_train_name_r << endl;
			for (int i = 0; i <more_than_least; i++)
			{
				for (int j = 0; j < labels_train[i][n_set + 1]; j++)
				{
					for (int k = 0; k < feature_size; k++)
					{
						finfeature_train << X[(labels_train[i][j + 1] - 1)][k] << "\t";
					}
					finfeature_train << endl;
				}
			}
		}
		finfeature_train.close();
		
		
		string feature_test_name_r = feature_test_name;
		fstream finfeature_test;
		finfeature_test.open(feature_test_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finfeature_test)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "can not read:" << feature_test_name_r << endl;
		}
		else
		{
			cout << "read:" << feature_test_name_r << endl;
			for (int i = 0; i < n_test; i++)
			{
				for (int j = 0; j < one_persion_pick; j++)
				{
					for (int k = 0; k < feature_size; k++)
					{
						finfeature_test << X[(labels_test[i][j + 1] - 1)][k] << "\t";
					}
					finfeature_test << endl;
				}
			}
		}
		finfeature_test.close();
		cout << "train and test data choose be done" << endl;
		
	}
	else
	{
		cout << "pls check file" << endl;
	}
	
}
void jointbayesain_PCA(const char* labels_name, const char* feature_name, const char* feature_pca_name, const char* feature_mean_name, const char* pca_matrix_name)
{
	cout << "PCA start" << endl;
	vector < int > labels;
	bool check = 1;
	string labels_name_r = labels_name;
	ifstream finlabels(labels_name_r);
	int lebals_data_r;
	if (!finlabels) {
		cout << "can not read"<<labels_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << labels_name_r << endl;
		while (!finlabels.eof())
		{
			finlabels >> lebals_data_r;
			labels.push_back(lebals_data_r);
			finlabels.get();
			if (finlabels.peek() == '\n')
			{
				break;
			}
		}
	}
	finlabels.close();
	cout << "total class:" << labels.size() << endl;
	MatrixXf X(labels.size(), feature_size);
	X = MatrixXf::Zero(labels.size(), feature_size);
	string feature_name_r = feature_name;
	ifstream finfeature(feature_name_r);
	if (!finfeature) {
		cout << "can not read" << feature_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << feature_name_r << endl;
		for (unsigned int i = 0; i < labels.size(); i++)
		{
			for (unsigned int j = 0; j < feature_size; j++)
			{
				finfeature >> X(i, j);
			}
		}
	}
	finfeature.close();
	if (check == 1 && feature_size >= feature_pca_size)
	{
		RowVectorXf X_mean(feature_size);
		X_mean = X.colwise().mean();
		X=X.rowwise() - X_mean;
		MatrixXf X2 = X.transpose()*X;
		EigenSolver<MatrixXf> es_solve(X2);
		VectorXf es_solved_val = es_solve.eigenvalues().real();
		MatrixXf es_solved_vec = es_solve.eigenvectors().real();
		vector<vector<float>> eignevalue_location(2, vector<float>(feature_size));
		for (unsigned int i = 0; i < feature_size; i++)
		{
			eignevalue_location[0][i] = es_solved_val(i);
			eignevalue_location[1][i] = i;
		}
		int minIndex;
		for (unsigned int i = 0; i < feature_size; i++)
		{
			minIndex = i;
			for (unsigned int j = i + 1; j < feature_size; j++)
			{
				if (eignevalue_location[0][minIndex] < eignevalue_location[0][j])
				{
					minIndex = j;
				}
			}
			if (minIndex != i)
			{
				vector<float> temp{ eignevalue_location[0][i], eignevalue_location[1][i] };
				eignevalue_location[0][i] = eignevalue_location[0][minIndex];
				eignevalue_location[1][i] = eignevalue_location[1][minIndex];
				eignevalue_location[0][minIndex] = temp[0];
				eignevalue_location[1][minIndex] = temp[1];
			}
		}
		MatrixXf eigenvector_need(feature_size, feature_pca_size);
		for (unsigned int i = 0; i < feature_pca_size; i++)
		{
			eigenvector_need.col(i) = es_solved_vec.col((int)eignevalue_location[1][i]);
		}
		string pca_matrix_name_r = pca_matrix_name;
		fstream finpca_matrix;
		finpca_matrix.open(pca_matrix_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finpca_matrix)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "can not read" << pca_matrix_name_r << endl;
		}
		else
		{
			cout << "read" << pca_matrix_name_r << endl;
			//cout << eigenvector_need << endl;
			for (unsigned int i = 0; i <feature_size; i++)
			{
				for (unsigned int j = 0; j < feature_pca_size; j++)
				{
					finpca_matrix << eigenvector_need(i, j) << "\t";
				}
				finpca_matrix << endl;
			}
		}
		finpca_matrix.close();
		MatrixXf X_after_PCA = X*eigenvector_need;
		string feature_pca_name_r = feature_pca_name;
		fstream finfeature_pca;
		finfeature_pca.open(feature_pca_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finfeature_pca)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "µLªkŒg€J" << feature_pca_name_r << endl;
		}
		else
		{
			cout << "read" << feature_pca_name_r << endl;
			for (unsigned int i = 0; i <labels.size(); i++)
			{
				for (unsigned int j = 0; j < feature_pca_size; j++)
				{
					finfeature_pca << X_after_PCA(i, j) << "\t";
				}
				finfeature_pca << endl;
			}
		}
		finfeature_pca.close();
		string feature_mean_name_r = feature_mean_name;
		fstream finfeature_mean;
		finfeature_mean.open(feature_mean_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finfeature_mean)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "can not read" << feature_mean_name_r << endl;
		}
		else
		{
			cout << "read" << feature_mean_name_r << endl;
			for (unsigned int m = 0; m < feature_size; m++)
			{
				finfeature_mean << X_mean(m) << "\t";
			}
		}
		finfeature_mean.close();
		cout << "PCA over" << endl;
	}
	else if (check == 0)
	{
		cout << "pls check file" << endl;
	}
	else if (feature_size < feature_pca_size)
	{
		cout << "pls check PCA dimension" << endl;
	}

}
void jointbayesain_AG(const char* labels_name, const char* feature_name, const char* A_name, const char* G_name)
{
	cout << "A G­pºâ¶}©l" << endl;
	int convergence_count = 0;
	clock_t start_t2, end_t2;
	float convergence_set = 0.0000006;
	unsigned int loop_set = 500;
	vector < int > labels;
	bool check = 1;
	string labels_name_r = labels_name;
	ifstream finlabels(labels_name_r);
	int lebals_data_r;
	if (!finlabels) {
		cout << "µLªkÅª€J" << labels_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "Åª€J" << labels_name_r << endl;
		while (!finlabels.eof())
		{
			finlabels >> lebals_data_r;
			labels.push_back(lebals_data_r);
			finlabels.get();
			if (finlabels.peek() == '\n')
			{
				break;
			}
		}
	}
	finlabels.close();
	cout << "A matrix start:" << labels.size() << endl;
	vector<vector<float>> X(labels.size(), vector<float>(feature_pca_size));
	string feature_name_r = feature_name;
	ifstream finfeature(feature_name_r);
	if (!finfeature) {
		cout << "can not read" << feature_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << feature_name_r << endl;
		for (unsigned int i = 0; i < labels.size(); i++)
		{
			for (unsigned int j = 0; j < feature_pca_size; j++)
			{
				finfeature >> X[i][j];
			}
		}
	}
	finfeature.close();
	if (check == 1)
	{
		vector < int > numberBuff;
		vector < int > labels_class(labels);
		sort(labels_class.begin(), labels_class.end());
		vector<int>::iterator iter = unique(labels_class.begin(), labels_class.end());
		labels_class.erase(iter, labels_class.end());
		vector<vector<int>> labels_location(labels_class.size(), vector<int>(n_set + 1));
		for (unsigned int i = 0; i < labels_class.size(); i++)
		{
			unsigned int k = 0;
			for (unsigned int j = 0; j < labels.size(); j++)
			{
				if (labels_class[i] == labels[j])
				{
					labels_location[i][k] = j + 1;//labelsŠìžm¬ö¿ý,žò¹s°Ï§O,¹ê»Ú­n¥Î®ÉŽî1
					if (k < n_set)
					{
						k = k + 1;
					}
				}
				labels_location[i][n_set] = k;//¬ö¿ýšC€HŠ³ŽX±i
			}
			vector<int>::iterator found;
			found = find(numberBuff.begin(), numberBuff.end(), k);
			if (found == numberBuff.end())
			{
				numberBuff.push_back(k);
			}
		}
		sort(numberBuff.begin(), numberBuff.end());
		MatrixXf ep(labels.size(), feature_pca_size);
		ep = MatrixXf::Zero(labels.size(), feature_pca_size);
		MatrixXf u(labels_class.size(), feature_pca_size);
		u = MatrixXf::Zero(labels_class.size(), feature_pca_size);
		for (unsigned int i = 0; i < labels_class.size(); i++)
		{
			for (unsigned int k = 0; k < feature_pca_size; k++)
			{
				for (unsigned int j = 0; j < labels_location[i][n_set]; j++)
				{
					u(i, k) = u(i, k) + X[(labels_location[i][j] - 1)][k];
				}
				u(i, k) = u(i, k) / labels_location[i][n_set];
			}
		}
		for (unsigned int i = 0; i < labels_class.size(); i++)
		{
			for (unsigned int j = 0; j < labels_location[i][n_set]; j++)
			{
				for (unsigned int k = 0; k < feature_pca_size; k++)
				{
					ep((labels_location[i][j] - 1), k) = X[(labels_location[i][j] - 1)][k] - u(i, k);
				}
			}
		}
		cout << "classification over" << endl;
		MatrixXf centered1 = ep.rowwise() - ep.colwise().mean();
		MatrixXf Sw = (centered1.adjoint() * centered1) / float(ep.rows() - 1);//covariance of ep_transpose
		MatrixXf Sw_old = Sw;
		MatrixXf centered2 = u.rowwise() - u.colwise().mean();
		MatrixXf Su = (centered2.adjoint() * centered2) / float(u.rows() - 1);//covariance of u_transpose
		MatrixXf F(feature_pca_size, feature_pca_size);
		MatrixXf G(feature_pca_size, feature_pca_size);
		MatrixXf gSu_pluse_Sw(feature_pca_size, feature_pca_size);
		vector<MatrixXf> SuFG(numberBuff.size(), MatrixXf(feature_pca_size, feature_pca_size));
		vector<MatrixXf> SwG(numberBuff.size(), MatrixXf(feature_pca_size, feature_pca_size));
		for (unsigned int i = 0; i < numberBuff.size(); i++)
		{
			SuFG[i] = MatrixXf::Zero(feature_pca_size, feature_pca_size);
			SwG[i] = MatrixXf::Zero(feature_pca_size, feature_pca_size);
		}
		for (unsigned int i = 0; i < loop_set; i++)
		{
			start_t2 = clock();
			F = Sw.inverse();
			ep = MatrixXf::Zero(labels.size(), feature_pca_size);
			for (unsigned int j = 0; j < numberBuff.size(); j++)
			{
				gSu_pluse_Sw = numberBuff[j] * Su + Sw;
				G = -1 * gSu_pluse_Sw.inverse()*Su*Sw.inverse();
				SuFG[j] = Su*(F + numberBuff[j] * G);
				SwG[j] = Sw*G;
			}
			for (unsigned int k = 0; k < labels_class.size(); k++)
			{
				int ncc = labels_location[k][n_set];
				for (unsigned int l = 0; l < numberBuff.size(); l++)
				{
					if (numberBuff[l] == ncc)
					{
						ncc = l;
						break;
					}
				}
				MatrixXf cur(numberBuff[ncc], feature_pca_size);
				for (unsigned int m = 0; m < numberBuff[ncc]; m++)
				{
					for (unsigned int n = 0; n < feature_pca_size; n++)
					{
						cur(m, n) = X[(labels_location[k][m]) - 1][n];
					}
				}
				MatrixXf curSuFG = cur*SuFG[ncc].transpose();
				MatrixXf curSwG = cur*SwG[ncc].transpose();
				vector<float> curSwG_sum(feature_pca_size, 0);
				for (unsigned int n = 0; n < feature_pca_size; n++)
				{
					float curSuFG_sum_cal = 0;
					float curSwG_sum_cal = 0;
					for (unsigned int m = 0; m < numberBuff[ncc]; m++)
					{
						curSuFG_sum_cal = curSuFG_sum_cal + curSuFG(m, n);
						curSwG_sum_cal = curSwG_sum_cal + curSwG(m, n);
					}
					u(k, n) = curSuFG_sum_cal;
					curSwG_sum[n] = curSwG_sum_cal;
				}
				for (unsigned int m = 0; m < numberBuff[ncc]; m++)
				{
					for (unsigned int n = 0; n < feature_pca_size; n++)
					{
						ep(((labels_location[k][m]) - 1), n) = X[((labels_location[k][m]) - 1)][n] + curSwG_sum[n];
					}
				}
			}

			centered1 = ep.rowwise() - ep.colwise().mean();
			Sw = (centered1.adjoint() * centered1) / float(ep.rows() - 1);//covariance of ep_transpose
			centered2 = u.rowwise() - u.colwise().mean();
			Su = (centered2.adjoint() * centered2) / float(u.rows() - 1);//covariance of u_transpose
			float check_convergence = (Sw - Sw_old).norm() / Sw.norm();
			cout << i << " " << check_convergence << endl;
			if (check_convergence <  convergence_set)
			{
				convergence_count = convergence_count + 1;
				//break;
			}
			else
			{
				convergence_count = 0;
			}
			if (convergence_count >= 10)
			{
				break;
			}
			Sw_old = Sw;
			end_t2 = clock();
			cout << "total loop time: " << (end_t2 - start_t2) << "ms" << endl;
		}
		G = -1 * (2 * Su + Sw).inverse()*Su*Sw.inverse();
		MatrixXf A = (Su + Sw).inverse() - (Sw.inverse() + G);
		string A_name_r = A_name;
		fstream finA;
		finA.open(A_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finA)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "can not read" << A_name_r << endl;
		}
		else
		{
			cout << "read" << A_name_r << endl;
			for (unsigned int i = 0; i <feature_pca_size; i++)
			{
				for (unsigned int j = 0; j < feature_pca_size; j++)
				{
					finA << fixed << setprecision(6) << A(i, j) << "\t";
				}
				finA << endl;
			}
		}
		finA.close();
		string G_name_r = G_name;
		fstream finG;
		finG.open(G_name_r, ios::out);//¶}±ÒÀÉ®×
		if (!finG)
		{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
			cout << "can not read" << G_name_r << endl;
		}
		else
		{
			cout << "read" << G_name_r << endl;
			for (unsigned int i = 0; i < feature_pca_size; i++)
			{
				for (unsigned int j = 0; j < feature_pca_size; j++)
				{
					finG << fixed << setprecision(6) << G(i, j) << "\t";
				}
				finG << endl;
			}
		}
		finG.close();
		cout << "A, G calculation over" << endl;
	}
	else
	{
		cout << "pls check file" << endl;
	}
}
void jointbayesain_ratio(const char* labels_test_name, const char* feature_test_name, const char* feature_mean_name, const char* pca_matrix_name, const char* A_name, const char* G_name, const char* postivepair_ratio_name, const char* nagtivepair_ratio_name)
{
	cout << "start test data" << endl;
	unsigned int Permutations = (one_persion_pick*(one_persion_pick - 1)) / 2;
	vector<vector<int>> C_n_get_2(Permutations, vector<int>(2));
	
	int num = 0;
	for (unsigned int i = 0; i < one_persion_pick - 1; i++)
	{
		for (unsigned int j = i + 1; j < one_persion_pick; j++)
		{
			C_n_get_2[num][0] = i;
			C_n_get_2[num][1] = j;
			num = num + 1;
		}
	}

	unsigned int Permutations_nagtivepair = (n_test*(n_test - 1)) / 2;
	vector<vector<int>> C_n_get_2_nagtivepair(Permutations_nagtivepair, vector<int>(2));

	num = 0;
	for (unsigned int i = 0; i < n_test - 1; i++)
	{
		for (unsigned int j = i + 1; j < n_test; j++)
		{
			C_n_get_2_nagtivepair[num][0] = i;
			C_n_get_2_nagtivepair[num][1] = j;
			num = num + 1;
		}
	}

	vector < int > labels;
	bool check = 1;
	string labels_test_name_r = labels_test_name;
	ifstream finlabels(labels_test_name_r);
	int lebals_data_r;
	if (!finlabels) {
		cout << "can not read" << labels_test_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << labels_test_name_r << endl;
		while (!finlabels.eof())
		{
			finlabels >> lebals_data_r;
			labels.push_back(lebals_data_r);
			finlabels.get();
			if (finlabels.peek() == '\n')
			{
				break;
			}
		}
	}
	finlabels.close();
	cout << "total class:" << labels.size() << endl;
	if ((labels.size() % 2) != 0)
	{
		check = 0;
	}
	MatrixXf X = MatrixXf::Zero(labels.size(), feature_size);
	string feature_test_name_r = feature_test_name;
	ifstream finfeature(feature_test_name_r);
	if (!finfeature) {
		cout << "can not read" << feature_test_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << feature_test_name_r << endl;
		for (unsigned int i = 0; i < labels.size(); i++)
		{
			for (unsigned int j = 0; j < feature_size; j++)
			{
				finfeature >> X(i, j);
			}
		}
	}
	finfeature.close();
	RowVectorXf meanX = MatrixXf::Zero(1, feature_size);
	string feature_mean_name_r = feature_mean_name;
	ifstream finfeature_mean(feature_mean_name_r);
	if (!finfeature_mean) {
		cout << "can not read" << feature_mean_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << feature_mean_name_r << endl;
		for (unsigned int i = 0; i < feature_size; i++)
		{
			finfeature_mean >> meanX(i);
		}
	}
	finfeature_mean.close();
	MatrixXf eigenvector_need(feature_size, feature_pca_size);
	string pca_matrix_name_r = pca_matrix_name;
	ifstream fineigenvector_need(pca_matrix_name_r);
	if (!fineigenvector_need) {
		cout << "can not read" << pca_matrix_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << pca_matrix_name_r << endl;
		for (unsigned int i = 0; i < feature_size; i++)
		{
			for (unsigned int j = 0; j < feature_pca_size; j++)
			{
				fineigenvector_need >> eigenvector_need(i, j);
			}
		}
	}
	fineigenvector_need.close();
	MatrixXf A(feature_pca_size, feature_pca_size);
	string A_name_r = A_name;
	ifstream finA(A_name_r);
	if (!finA) {
		cout << "can not read" << A_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << A_name_r << endl;
		for (unsigned int i = 0; i < feature_pca_size; i++)
		{
			for (unsigned int j = 0; j < feature_pca_size; j++)
			{
				finA >> A(i, j);
			}
		}
	}
	finA.close();
	MatrixXf G(feature_pca_size, feature_pca_size);
	string G_name_r = G_name;
	ifstream finG(G_name_r);
	if (!finG) {
		cout << "can not read" << G_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "read" << G_name_r << endl;
		for (unsigned int i = 0; i < feature_pca_size; i++)
		{
			for (unsigned int j = 0; j < feature_pca_size; j++)
			{
				finG >> G(i, j);
			}
		}
	}
	finG.close();

	//
	if (check == 1)
	{
		MatrixXf X2 = (X.rowwise() - meanX)*eigenvector_need;
		vector<float> postivepair;
		vector<float> nagtivepair;
		vector<vector<int>> nagtivepair_record(Permutations_nagtivepair*one_persion_pick*one_persion_pick, vector<int>(4));
		vector<vector<int>> postivepair_record(Permutations*n_test, vector<int>(4));
		unsigned int choose1;
		unsigned int choose2;
		//unsigned int choose3;
		float ratio = 0;
		for (unsigned int i = 0; i < n_test; i++)
		{
			for (unsigned int j = 0; j < Permutations; j++)
			{

				//choose3 = ((i + 1)*one_persion_pick) + C_n_get_2[j][1];
				choose1=(i*one_persion_pick) + C_n_get_2[j][0];
				choose2=(i*one_persion_pick) + C_n_get_2[j][1];
				/*
				if (choose3 > one_persion_pick*n_test)
				{
					choose3 = choose3 - one_persion_pick*(n_test);
				}
				*/
				ratio = 0;
				ratio = ratio + (X2.row(choose1)*A*X2.row(choose1).transpose());
				ratio = ratio + (X2.row(choose2)*A*X2.row(choose2).transpose());
				ratio = ratio - (X2.row(choose1)*G*X2.row(choose2).transpose());
				ratio = ratio - (X2.row(choose1)*G*X2.row(choose2).transpose());
				postivepair.push_back(ratio);
				postivepair_record[postivepair.size()-1][0] = labels[choose1];
				postivepair_record[postivepair.size()-1][1] = labels[choose2];
				postivepair_record[postivepair.size() - 1][2] = test_record(labels[choose1], C_n_get_2[j][0] + 1);
				postivepair_record[postivepair.size() - 1][3] = test_record(labels[choose2], C_n_get_2[j][1] + 1);

				/*
				ratio = 0;
				ratio = ratio + (X2.row(choose1)*A*X2.row(choose3).transpose());
				ratio = ratio + (X2.row(choose3)*A*X2.row(choose3).transpose());
				ratio = ratio - (X2.row(choose1)*G*X2.row(choose3).transpose());
				ratio = ratio - (X2.row(choose1)*G*X2.row(choose3).transpose());
				nagtivepair.push_back(ratio);
				nagtivepair_record[nagtivepair.size() - 1][0] = labels[choose1];
				nagtivepair_record[nagtivepair.size() - 1][1] = labels[choose3];
				nagtivepair_record[nagtivepair.size() - 1][2] = test_record(labels[choose1], C_n_get_2[j][0] + 1);
				nagtivepair_record[nagtivepair.size() - 1][3] = test_record(labels[choose3], C_n_get_2[j][1] + 1);
				*/

			}
		}
		for (unsigned int i = 0; i < Permutations_nagtivepair; i++)
		{
			for (unsigned int j = 0; j < one_persion_pick; j++)
			{
				for (unsigned int k = 0; k < one_persion_pick; k++)
				{
					choose1 = C_n_get_2_nagtivepair[i][0] * one_persion_pick + j;
					choose2 = C_n_get_2_nagtivepair[i][1] * one_persion_pick + k;
					ratio = 0;
					ratio = ratio + (X2.row(choose1)*A*X2.row(choose1).transpose());
					ratio = ratio + (X2.row(choose2)*A*X2.row(choose2).transpose());
					ratio = ratio - (X2.row(choose1)*G*X2.row(choose2).transpose());
					ratio = ratio - (X2.row(choose1)*G*X2.row(choose2).transpose());
					nagtivepair.push_back(ratio);
					nagtivepair_record[nagtivepair.size() - 1][0] = labels[choose1];
					nagtivepair_record[nagtivepair.size() - 1][1] = labels[choose2];
					nagtivepair_record[nagtivepair.size() - 1][2] = test_record(labels[choose1], j + 1);
					nagtivepair_record[nagtivepair.size() - 1][3] = test_record(labels[choose2], k + 1);
					//cout <<"ratio ­pºâŠÊ€À€ñ=" <<float(nagtivepair.size() *100)/ (Permutations_nagtivepair*one_persion_pick*one_persion_pick)<< "%"<< endl;

				}
			}
		}

		float ratio_max = *(max_element(begin(postivepair), end(postivepair)));
		float ratio_min = *(min_element(begin(nagtivepair), end(nagtivepair)));
		vector<float> TPR, FAR;
		if (ratio_max <= ratio_min )
		{
			cout << "data has problem!" << endl;
		}
		else
		{
			string postivepair_ratio_name_r = postivepair_ratio_name;
			fstream finratio_p;
			finratio_p.open(postivepair_ratio_name_r, ios::out);//¶}±ÒÀÉ®×
			if (!finratio_p)
			{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
				cout << "can not read" << postivepair_ratio_name_r << endl;
			}
			else
			{
				cout << "read" << postivepair_ratio_name_r << endl;
				for (unsigned int i = 0; i < postivepair.size(); i++)
				{
					finratio_p << fixed << setprecision(6) << postivepair[i] << "\t" << postivepair_record[i][0] << "\t" << postivepair_record[i][1] << "\t" << postivepair_record[i][2] << "\t" << postivepair_record[i][3]  << endl;
				}
			}
			finratio_p.close();

			for (unsigned int j = 0; j <= int(nagtivepair.size()/1000000); j++)
			{
				string nagtivepair_ratio_name_r = nagtivepair_ratio_name + to_string(j)+".txt";
				fstream finratio_n;
				finratio_n.open(nagtivepair_ratio_name_r, ios::out);//¶}±ÒÀÉ®×
				if (!finratio_n)
				{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
					cout << "can not read" << nagtivepair_ratio_name_r << endl;
				}
				else
				{
					cout << "read" << nagtivepair_ratio_name_r << endl;
					if ((j + 1 )* 1000000 <= nagtivepair.size())
					{
						for (unsigned int i = j * 1000000; i < ((j + 1) * 1000000); i++)
						{
							finratio_n << fixed << setprecision(6) << nagtivepair[i] << "\t" << nagtivepair_record[i][0] << "\t" << nagtivepair_record[i][1] << "\t" << nagtivepair_record[i][2] << "\t" << nagtivepair_record[i][3] << endl;
						}
					}
					else
					{
						for (unsigned int i = j * 1000000; i < nagtivepair.size(); i++)
						{
							finratio_n << fixed << setprecision(6) << nagtivepair[i] << "\t" << nagtivepair_record[i][0] << "\t" << nagtivepair_record[i][1] << "\t" << nagtivepair_record[i][2] << "\t" << nagtivepair_record[i][3] << endl;
						}
					}
					
				}
				finratio_n.close();
			}
			/*
			string nagtivepair_ratio_name_r = nagtivepair_ratio_name;
			fstream finratio_n;
			finratio_n.open(nagtivepair_ratio_name_r, ios::out);//¶}±ÒÀÉ®×
			if (!finratio_n)
			{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
				cout << "µLªkŒg€J" << nagtivepair_ratio_name_r << endl;
			}
			else
			{
				cout << "Œg€J" << nagtivepair_ratio_name_r << endl;
				for (unsigned int i = 0; i < nagtivepair.size(); i++)
				{
					finratio_n << fixed << setprecision(6) << nagtivepair[i] << "\t" << nagtivepair_record[i][0] << "\t" << nagtivepair_record[i][1] << "\t" << nagtivepair_record[i][2] << "\t" << nagtivepair_record[i][3] << endl;
				}
			}
			finratio_n.close();
			*/

			for (float threshold = ratio_min - 1; threshold <= ratio_max + 1; threshold += ((ratio_max - ratio_min + 2) / 6000))
			{
				TPR.push_back((float)count_if(postivepair.begin(), postivepair.end(), bind2nd(greater<float>(), threshold)) / postivepair.size());
				FAR.push_back(1 - ((float)count_if(nagtivepair.begin(), nagtivepair.end(), bind2nd(less<float>(), threshold)) / nagtivepair.size()));
			}
			/*
			string ROC_name_r = ROC_name;
			fstream finROC;
			finROC.open(ROC_name_r, ios::out);//¶}±ÒÀÉ®×
			if (!finROC)
			{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
				cout << "µLªkŒg€J" << ROC_name_r << endl;
			}
			else
			{
				cout << "Œg€J" << ROC_name_r << endl;
				for (unsigned int i = 0; i < TPR.size(); i++)
				{
					finROC << fixed << setprecision(6) << FAR[i] << "\t" << TPR[i] << endl;
				}
			}
			finROC.close();
			*/
			float sum_cal = 0;
			for (unsigned int i = TPR.size()-1; i >0; i--)
			{
				sum_cal = (FAR[i-1] - FAR[i])*TPR[i] + sum_cal;
			}
			cout << "ROC=" << sum_cal << endl;
			cout << "test data over" << endl;
		}
	}
	else
	{
		cout << "pls check file" << endl;
	}
}

void jointbayesain_ratio_old(const char* labels_test_name, const char* feature_test_name, const char* feature_mean_name, const char* pca_matrix_name, const char* A_name, const char* G_name, const char* ratio_name, const char* ROC_name)
{
	cout << "¶}©lŽúžÕteat data" << endl;
	unsigned int Permutations = (one_persion_pick*(one_persion_pick - 1)) / 2;
	vector<vector<int>> C_n_get_2(Permutations, vector<int>(2));

	int num = 0;
	for (unsigned int i = 0; i < one_persion_pick - 1; i++)
	{
		for (unsigned int j = i + 1; j < one_persion_pick; j++)
		{
			C_n_get_2[num][0] = i;
			C_n_get_2[num][1] = j;
			num = num + 1;
		}
	}

	vector < int > labels;
	bool check = 1;
	string labels_test_name_r = labels_test_name;
	ifstream finlabels(labels_test_name_r);
	int lebals_data_r;
	if (!finlabels) {
		cout << "µLªkÅª€J" << labels_test_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "Åª€J" << labels_test_name_r << endl;
		while (!finlabels.eof())
		{
			finlabels >> lebals_data_r;
			labels.push_back(lebals_data_r);
			finlabels.get();
			if (finlabels.peek() == '\n')
			{
				break;
			}
		}
	}
	finlabels.close();
	cout << "Á`žê®Æ€HŒÆ:" << labels.size() << endl;
	if ((labels.size() % 2) != 0)
	{
		check = 0;
	}
	MatrixXf X = MatrixXf::Zero(labels.size(), feature_size);
	string feature_test_name_r = feature_test_name;
	ifstream finfeature(feature_test_name_r);
	if (!finfeature) {
		cout << "µLªkÅª€J" << feature_test_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "Åª€J" << feature_test_name_r << endl;
		for (unsigned int i = 0; i < labels.size(); i++)
		{
			for (unsigned int j = 0; j < feature_size; j++)
			{
				finfeature >> X(i, j);
			}
		}
	}
	finfeature.close();
	RowVectorXf meanX = MatrixXf::Zero(1, feature_size);
	string feature_mean_name_r = feature_mean_name;
	ifstream finfeature_mean(feature_mean_name_r);
	if (!finfeature_mean) {
		cout << "µLªkÅª€J" << feature_mean_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "Åª€J" << feature_mean_name_r << endl;
		for (unsigned int i = 0; i < feature_size; i++)
		{
			finfeature_mean >> meanX(i);
		}
	}
	finfeature_mean.close();
	MatrixXf eigenvector_need(feature_size, feature_pca_size);
	string pca_matrix_name_r = pca_matrix_name;
	ifstream fineigenvector_need(pca_matrix_name_r);
	if (!fineigenvector_need) {
		cout << "µLªkÅª€J" << pca_matrix_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "Åª€J" << pca_matrix_name_r << endl;
		for (unsigned int i = 0; i < feature_size; i++)
		{
			for (unsigned int j = 0; j < feature_pca_size; j++)
			{
				fineigenvector_need >> eigenvector_need(i, j);
			}
		}
	}
	fineigenvector_need.close();
	MatrixXf A(feature_pca_size, feature_pca_size);
	string A_name_r = A_name;
	ifstream finA(A_name_r);
	if (!finA) {
		cout << "µLªkÅª€J" << A_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "Åª€J" << A_name_r << endl;
		for (unsigned int i = 0; i < feature_pca_size; i++)
		{
			for (unsigned int j = 0; j < feature_pca_size; j++)
			{
				finA >> A(i, j);
			}
		}
	}
	finA.close();
	MatrixXf G(feature_pca_size, feature_pca_size);
	string G_name_r = G_name;
	ifstream finG(G_name_r);
	if (!finG) {
		cout << "µLªkÅª€J" << G_name_r << endl;
		check = 0;
	}
	else
	{
		cout << "Åª€J" << G_name_r << endl;
		for (unsigned int i = 0; i < feature_pca_size; i++)
		{
			for (unsigned int j = 0; j < feature_pca_size; j++)
			{
				finG >> G(i, j);
			}
		}
	}
	finG.close();

	//
	if (check == 1)
	{
		MatrixXf X2 = (X.rowwise() - meanX)*eigenvector_need;
		vector<float> postivepair;
		vector<float> nagtivepair;
		vector<vector<int>> nagtivepair_record(Permutations*n_test, vector<int>(4));
		vector<vector<int>> postivepair_record(Permutations*n_test, vector<int>(4));
		unsigned int choose1;
		unsigned int choose2;
		unsigned int choose3;
		float ratio = 0;
		for (unsigned int i = 0; i < n_test; i++)
		{
			for (unsigned int j = 0; j < Permutations; j++)
			{
				/*
				int nagtive_class = rand() % n_test;
				if (nagtive_class == 0)
				{
				nagtive_class = 1;
				}
				choose3 = ((i + nagtive_class)*one_persion_pick) + C_n_get_2[j][1];
				*/
				choose3 = ((i + 1)*one_persion_pick) + C_n_get_2[j][1];
				choose1 = (i*one_persion_pick) + C_n_get_2[j][0];
				choose2 = (i*one_persion_pick) + C_n_get_2[j][1];
				if (choose3 > one_persion_pick*n_test)
				{
					choose3 = choose3 - one_persion_pick*(n_test);
				}
				ratio = 0;
				ratio = ratio + (X2.row(choose1)*A*X2.row(choose1).transpose());
				ratio = ratio + (X2.row(choose2)*A*X2.row(choose2).transpose());
				ratio = ratio - (X2.row(choose1)*G*X2.row(choose2).transpose());
				ratio = ratio - (X2.row(choose1)*G*X2.row(choose2).transpose());
				postivepair.push_back(ratio);
				postivepair_record[postivepair.size() - 1][0] = labels[choose1];
				postivepair_record[postivepair.size() - 1][1] = labels[choose2];
				postivepair_record[postivepair.size() - 1][2] = test_record(labels[choose1], C_n_get_2[j][0] + 1);
				postivepair_record[postivepair.size() - 1][3] = test_record(labels[choose2], C_n_get_2[j][1] + 1);
				//postivepair_record[postivepair.size()-1][2] = C_n_get_2[j][0];
				//postivepair_record[postivepair.size()-1][3] = C_n_get_2[j][1];

				ratio = 0;
				ratio = ratio + (X2.row(choose1)*A*X2.row(choose3).transpose());
				ratio = ratio + (X2.row(choose3)*A*X2.row(choose3).transpose());
				ratio = ratio - (X2.row(choose1)*G*X2.row(choose3).transpose());
				ratio = ratio - (X2.row(choose1)*G*X2.row(choose3).transpose());
				nagtivepair.push_back(ratio);
				nagtivepair_record[nagtivepair.size() - 1][0] = labels[choose1];
				nagtivepair_record[nagtivepair.size() - 1][1] = labels[choose3];
				nagtivepair_record[nagtivepair.size() - 1][2] = test_record(labels[choose1], C_n_get_2[j][0] + 1);
				nagtivepair_record[nagtivepair.size() - 1][3] = test_record(labels[choose3], C_n_get_2[j][1] + 1);
				//nagtivepair_record[nagtivepair.size() - 1][2] = C_n_get_2[j][0];
				//nagtivepair_record[nagtivepair.size() - 1][3] = C_n_get_2[j][1];
			}
		}

		float ratio_max = *(max_element(begin(postivepair), end(postivepair)));
		float ratio_min = *(min_element(begin(nagtivepair), end(nagtivepair)));
		vector<float> TPR, FAR;
		if (ratio_max <= ratio_min || nagtivepair.size() != nagtivepair.size())
		{
			cout << "žê®ÆŠ³°ÝÃD" << endl;
		}
		else
		{
			string ratio_name_r = ratio_name;
			fstream finratio;
			finratio.open(ratio_name_r, ios::out);//¶}±ÒÀÉ®×
			if (!finratio)
			{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
				cout << "µLªkŒg€J" << ratio_name_r << endl;
			}
			else
			{
				cout << "Œg€J" << ratio_name_r << endl;
				for (unsigned int i = 0; i < postivepair.size(); i++)
				{
					finratio << fixed << setprecision(6) << postivepair[i] << "\t" << postivepair_record[i][0] << "\t" << postivepair_record[i][1] << "\t" << postivepair_record[i][2] << "\t" << postivepair_record[i][3] << "\t" << nagtivepair[i] << "\t" << nagtivepair_record[i][0] << "\t" << nagtivepair_record[i][1] << "\t" << nagtivepair_record[i][2] << "\t" << nagtivepair_record[i][3] << endl;
				}
			}
			finratio.close();
			for (float threshold = ratio_min - 1; threshold <= ratio_max + 1; threshold += ((ratio_max - ratio_min + 2) / 6000))
			{
				TPR.push_back((float)count_if(postivepair.begin(), postivepair.end(), bind2nd(greater<float>(), threshold)) / postivepair.size());
				FAR.push_back(1 - ((float)count_if(nagtivepair.begin(), nagtivepair.end(), bind2nd(less<float>(), threshold)) / nagtivepair.size()));
			}
			string ROC_name_r = ROC_name;
			fstream finROC;
			finROC.open(ROC_name_r, ios::out);//¶}±ÒÀÉ®×
			if (!finROC)
			{//ŠpªG¶}±ÒÀÉ®×¥¢±Ñ¡Afp¬°0¡FŠš¥\¡Afp¬°«D0
				cout << "µLªkŒg€J" << ROC_name_r << endl;
			}
			else
			{
				cout << "Œg€J" << ROC_name_r << endl;
				for (unsigned int i = 0; i < TPR.size(); i++)
				{
					finROC << fixed << setprecision(6) << FAR[i] << "\t" << TPR[i] << endl;
				}
			}
			finROC.close();
			float sum_cal = 0;
			for (unsigned int i = TPR.size() - 1; i >0; i--)
			{
				sum_cal = (FAR[i - 1] - FAR[i])*TPR[i] + sum_cal;
			}
			cout << "ROC=" << sum_cal << endl;
			cout << "test dataµ²§ô" << endl;
		}
	}
	else
	{
		cout << "œÐœT»{ÀÉ®×" << endl;
	}
}







//===================================================
/*
#include <iostream>
#include <Eigen/Dense>


//using Eigen::MatrixXd;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;


using namespace std;


int main()
{

        cout<<"*******************1D-object****************"<<endl;


        Vector4d v1;
        v1<< 1,2,3,4;
        cout<<"v1=\n"<<v1<<endl;


        VectorXd v2(3);
        v2<<1,2,3;
        cout<<"v2=\n"<<v2<<endl;


        Array4i v3;
        v3<<1,2,3,4;
        cout<<"v3=\n"<<v3<<endl;


        ArrayXf v4(3);
        v4<<1,2,3;
        cout<<"v4=\n"<<v4<<endl;
}
*/

