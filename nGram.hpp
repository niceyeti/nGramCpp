/*
  A very small word/string based ngram prediction model. The model uses U16 keys to represent words,
  rather than storing the strings themselves in the data structures that store sequential data.
  NOTE that this means supporting only training data with up to 65535 unique words!!!
  The current workaround is to delete words which occur only once, since pruning/deleting very unlikely words
  should not effect maximum likelihood prediction estimates: most predictions will be for somewhat common sequences,
  since that is the scoring basis. Thus, eliminating the words that occur one or less times just trims the tail of the
  word distribution, which is unlikely to interfere with predictions at the top of the results.
  A similar justification for pruning comes from the inverse of smoothing: smoothing strategies intend to lift the probability
  of word permutations not seen in the training data, which occur zero times, but are also roughly equal to one.
  Eliminating these should have little effect at the max-likelihood end of the predictions, where predictions coalesce.
*/

#include <list>
#include <map>
#include <unordered_set> //use these for result duplicate subkey filtering
#include <vector>
#include <iostream>
#include <fstream>
//#include <cctype>
//#include <sstream>
//#include <unistd.h>
//#include <sys/types.h>
//#include <fcntl.h>
//#include <sys/stat.h>
//#include <wait.h>
//#include <utility>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <cmath>
#include <sys/time.h>
#include <sys/resource.h>

//defines max foreseeable ligetSubEnne length in the freqTable.txt database
#define MAX_LINE_LEN 256
#define PERIOD_HOLDER '+'
#define NGRAM 4
#define MIN_MODEL_SIZE 100 //Minimum items sufficient to define an ngram model. This is arbitrary, for the sake of code error-checks.
#define READ_SZ 4095
#define BUFSIZE 4096
#define MAX_WORDS_PER_PHRASE 256  //these params are not very safe in updateNgramTable--possible segfaults
#define MAX_PHRASES_PER_READ 256
#define MAX_TOKENS_PER_READ 1024  //about bufsize / 4.
#define MAX_SENT_LEN 256  //not very robust. but a constraint is needed on the upperbound of sentence length, for phrase parsing data structures.
// avg sentence length is around 10-15 words, 20+ being a long sentence.
//#define PHRASE_DELIMITER '#'
//#define WORD_DELIMITER ' '
#define FILE_DELIMITER '|'
#define PERIOD_HOLDER '+'
#define ASCII_DELETE 127
#define INF_ENTROPY 9999  //constant for infinite entropy: 9999 bits is enormous (think of it as 2^9999) 
#define INF_PERPLEXITY 999999
#define NLAMBDASETS 8
#define NLAMBDAS 7
#define NGRAMS 4
#define MAX_WORD_LEN 27 //determined by looking up on the internet. There are english words over 28 chars, but very uncommon.
#define DBG 0
#define U16_MAX 65535
#define U32_MAX 4294967295

//using namespace std;
using std::cout;
using std::getline;
using std::endl;
using std::string;
using std::vector;
using std::cin;
using std::map;
//using std::multimap;
using std::unordered_set;
using std::list;
using std::sort;
using std::flush;
using std::pair;
using std::pow;
using std::fstream;
using std::ios;

//TODO: use/map these
enum tableIndices{ NIL, ONE_GRAM, TWO_GRAM, THREE_GRAM, FOUR_GRAM, FIVE_GRAM, SIX_GRAM };
enum testDataIndices{RAW_HITS, REAL_HITS, RAW_LAMBDA_HTS, REAL_LAMBDA_HITS};

//typedef unsigned long long U128;
typedef unsigned long int U64; //whether or not this is actually a 64-bit uint depends on architecture
typedef unsigned int U32; // may be U64, depending on sys
typedef unsigned short int U16;
typedef U16 IntKey;  //see header notes. This value determines the max number of unique words in the training data

//WARNING These data structures only work on 64 bit systems, and only supports up to four-gram sequences (each word gets a U16 key)
typedef map<U64,map<IntKey,double> > NgramTable;
typedef map<IntKey,double>::iterator InnerTableIt;
typedef NgramTable::iterator OuterTableIt;
typedef pair<InnerTableIt,double> ResultPair;
typedef list<ResultPair > ResultList;
typedef ResultList::iterator ResultListIt;



//key to string, and string to key manager data types
typedef map<IntKey,string> KeyStringMap;
typedef KeyStringMap::iterator KeyStringMapIt;
typedef map<string,IntKey> StringKeyMap;
typedef StringKeyMap::iterator StringKeyMapIt;

typedef struct lambdaSet{
  double l[NLAMBDAS];
  double boolAccuracy; //some hit counts are real-valued, instead of discrete. For instance, we may want to track if some result set contains the correct nextWord, though it is not the most likely word.
  double realAccuracy;
  double recall;         //recall tracks if next word is anywhere in result set: bool := nextWord in resultList[]
  double topSevenAccuracy; //tracks if nextWord is in the top seven results, a typical user-satisfaction window
  double nPredictions;
} LambdaSet;

typedef struct modelStat{
  double sumFrequency;
  double totalEntropy;               //raw entropy across a single model. Though seemingly meaningless for anything but 1-gram models, total entropy gives us a sparsity-measure for other n-gram models for n>1.
  double expectedSubEntropy;  //subentropy of a model is defined as the summation of entropy w/in each n-1 gram subset multiplied by its probability
  double meanSubEntropy;       //more or less meaningless, since its only a raw mean. expectedSubEntropy (an expected value) provides a more meaningful measure of subEntropy.
  double totalPerplexity;            //recall that by definition, perplexity is duplicate data, since perplexity = 2^(entropy(x)) for some x
  double expectedSubPerplexity;
  double booleanAccuracy;
  double realAccuracy;
} ModelStat;

class NgramModel{
  public:
    modelStat stats[5];  //index by ngram model number
    lambdaSet lambdas;

    string phraseDelimiters;
    string rawDelimiters;
    string wordDelimiters;
    string delimiters;
    char wordDelimiter;
    char phraseDelimiter;

    NgramTable unigramTable;
    NgramTable bigramTable;
    NgramTable trigramTable;
    NgramTable quadgramTable;

    //data structures for storing the actual words separately from their integer keys in the n-gram table
    IntKey idCounter;
    KeyStringMap KeyStringTable;
    StringKeyMap StringKeyTable;

    NgramModel();
    ~NgramModel();
    
    //interaction layer for the key/string model
    IntKey StringToKey(const string& word);
    bool KeyToString(IntKey key, string& str);
    bool AllocKey(const string& newWord, IntKey& key);
    
    //utils
    U64 MakeNgramModelKey(int model, IntKey w1, IntKey w2 = 0, IntKey w3 = 0);
    void UpdateNgramModel(NgramTable& table, U64 key, IntKey nextWord);
    void UpdateUnigramModel(NgramTable& unigrams, IntKey key);
    void TablesToLogSpace(void);
    void TableToLogSpace(NgramTable& table);
    void UnigramTableToLogSpace(NgramTable& unigrams);
    void WordToKeySequence(vector<string>& wordVec, vector<IntKey>& keySequence);
    void ScoreResult(IntKey actual, ResultList& results);
    void Predict(vector<IntKey> keySeq, int i, ResultList& results);
    void PruneSequence(vector<string>& wordVec);
    void NormalizeTables(void);
    void NormalizeUnigramTable(NgramTable& unitable);
    void NormalizeTable(NgramTable& table);
    double GetProb(int nModel, U64 key, U16 subkey);
    void PrintResults(void);
    U16 GetMax(NgramTable& table, U64 outerKey);
    void LambdaEM(void);

    //text processing
    void NormalizeText(char ibuf[BUFSIZE], string& ostr);
    void DelimitText(string& istr);
    bool IsWordDelimiter(char c);
    bool IsValidWord(const char* word);
    bool IsValidWord(const string& token);
    void ScrubHyphens(string& istr);
    void FinalPass(string& buf);
    void ToLower(string& myStr);
    void ToLower(char buf[BUFSIZE]);
    void RawPass(string& istr);
    bool IsDelimiter(const char c, const string& delims);
    void TextToWordSequence(const string& fname, vector<string>& wordVec);
    int Tokenize(char* ptrs[], char buf[BUFSIZE], const string& delims);
    bool IsPhraseDelimiter(char c);

    //public
    void Train(const string& fname);
    void Test(const string& fname);
};















