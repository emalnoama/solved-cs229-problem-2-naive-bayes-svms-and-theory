Download Link: https://assignmentchef.com/product/solved-cs229-problem-2-naive-bayes-svms-and-theory
<br>



<h1>1. Constructing kernels</h1>

In class, we saw that by choosing a kernel <em>K</em>(<em>x,z</em>) = <em>φ</em>(<em>x</em>)<em><sup>T </sup>φ</em>(<em>z</em>), we can implicitly map data to a high dimensional space, and have the SVM algorithm work in that space. One way to generate kernels is to explicitly define the mapping <em>φ </em>to a higher dimensional space, and then work out the corresponding <em>K</em>.

However in this question we are interested in direct construction of kernels. I.e., suppose we have a function <em>K</em>(<em>x,z</em>) that we think gives an appropriate similarity measure for our learning problem, and we are considering plugging <em>K </em>into the SVM as the kernel function. However for <em>K</em>(<em>x,z</em>) to be a valid kernel, it must correspond to an inner product in some higher dimensional space resulting from some feature mapping <em>φ</em>. Mercer’s theorem tells us that <em>K</em>(<em>x,z</em>) is a (Mercer) kernel if and only if for any finite set {<em>x</em><sup>(1)</sup><em>,…,x</em><sup>(<em>m</em>)</sup>}, the matrix <em>K </em>is symmetric and positive semidefinite, where the square matrix <em>K </em>∈ R<em><sup>m</sup></em><sup>×<em>m </em></sup>is given by <em>K<sub>ij </sub></em>= <em>K</em>(<em>x</em><sup>(<em>i</em>)</sup><em>,x</em><sup>(<em>j</em>)</sup>).

Now here comes the question: Let <em>K</em><sub>1</sub>, <em>K</em><sub>2 </sub>be kernels over R<em><sup>n</sup></em>×R<em><sup>n</sup></em>, let <em>a </em>∈ R<sup>+ </sup>be a positive real number, let <em>f </em>: R<em><sup>n </sup></em>7→ R be a real-valued function, let <em>φ </em>: R<em><sup>n </sup></em>→ R<em><sup>d </sup></em>be a function mapping from R<em><sup>n </sup></em>to R<em><sup>d</sup></em>, let <em>K</em><sub>3 </sub>be a kernel over R<em><sup>d </sup></em>× R<em><sup>d</sup></em>, and let <em>p</em>(<em>x</em>) a polynomial over <em>x </em>with <em>positive </em>coefficients.

For each of the functions <em>K </em>below, state whether it is necessarily a kernel. If you think it is, prove it; if you think it isn’t, give a counter-example.

<ul>

 <li><em>K</em>(<em>x,z</em>) = <em>K</em><sub>1</sub>(<em>x,z</em>) + <em>K</em><sub>2</sub>(<em>x,z</em>)</li>

 <li><em>K</em>(<em>x,z</em>) = <em>K</em><sub>1</sub>(<em>x,z</em>) − <em>K</em><sub>2</sub>(<em>x,z</em>)</li>

 <li><em>K</em>(<em>x,z</em>) = <em>aK</em><sub>1</sub>(<em>x,z</em>)</li>

 <li><em>K</em>(<em>x,z</em>) = −<em>aK</em><sub>1</sub>(<em>x,z</em>)</li>

 <li><em>K</em>(<em>x,z</em>) = <em>K</em><sub>1</sub>(<em>x,z</em>)<em>K</em><sub>2</sub>(<em>x,z</em>)</li>

 <li><em>K</em>(<em>x,z</em>) = <em>f</em>(<em>x</em>)<em>f</em>(<em>z</em>)</li>

 <li><em>K</em>(<em>x,z</em>) = <em>K</em><sub>3</sub>(<em>φ</em>(<em>x</em>)<em>,φ</em>(<em>z</em>))</li>

 <li><em>K</em>(<em>x,z</em>) = <em>p</em>(<em>K</em><sub>1</sub>(<em>x,z</em>))</li>

</ul>

[Hint: For part (e), the answer is that the <em>K </em>there <em>is </em>indeed a kernel. You still have to prove it, though. (This one may be harder than the rest.) This result may also be useful for another part of the problem.]

<h1>2.  Kernelizing the Perceptron</h1>

Let there be a binary classification problem with <em>y </em>∈ {0<em>,</em>1}. The perceptron uses hypotheses of the form <em>h<sub>θ</sub></em>(<em>x</em>) = <em>g</em>(<em>θ<sup>T </sup>x</em>), where <em>g</em>(<em>z</em>) = <strong>1</strong>{<em>z </em>≥ 0}. In this problem we will consider a stochastic gradient descent-like implementation of the perceptron algorithm where each update to the parameters <em>θ </em>is made using only one training example. However, unlike stochastic gradient descent, the perceptron algorithm will only make one pass through the entire training set. The update rule for this version of the perceptron algorithm is given by

<em>θ</em>(<em>i</em>+1) := <em>θ</em>(<em>i</em>) + <em>α</em>[<em>y</em>(<em>i</em>+1) − <em>h<sub>θ</sub></em>(<em>i</em>)(<em>x</em>(<em>i</em>+1))]<em>x</em>(<em>i</em>+1)

where <em>θ</em><sup>(<em>i</em>) </sup>is the value of the parameters after the algorithm has seen the first <em>i </em>training examples. Prior to seeing any training examples, <em>θ</em><sup>(0) </sup>is initialized to <em>~</em>0.

Let <em>K </em>be a Mercer kernel corresponding to some very high-dimensional feature mapping <em>φ</em>. Suppose <em>φ </em>is so high-dimensional (say, ∞-dimensional) that it’s infeasible to ever represent <em>φ</em>(<em>x</em>) explicitly. Describe how you would apply the “kernel trick” to the perceptron to make it work in the high-dimensional feature space <em>φ</em>, but without ever explicitly computing <em>φ</em>(<em>x</em>). [Note: You don’t have to worry about the intercept term. If you like, think of <em>φ </em>as having the property that <em>φ</em><sub>0</sub>(<em>x</em>) = 1 so that this is taken care of.] Your description should specify

<ul>

 <li>How you will (implicitly) represent the high-dimensional parameter vector <em>θ</em><sup>(<em>i</em>)</sup>, including how the initial value <em>θ</em><sup>(0) </sup>= <em>~</em>0 is represented (note that <em>θ</em><sup>(<em>i</em>) </sup>is now a vector whose dimension is the same as the feature vectors <em>φ</em>(<em>x</em>));</li>

 <li>How you will efficiently make a prediction on a new input <em>x</em><sup>(<em>i</em>+<a href="#_ftn1" name="_ftnref1">[1]</a>)</sup>. I.e., how you will compute <em>h<sub>θ</sub></em>(<em>i</em><sub>)</sub>(<em>x</em><sup>(<em>i</em>+1)</sup>) = <em>g</em>(<em>θ</em><sup>(<em>i</em>)<em>T </em></sup><em>φ</em>(<em>x</em><sup>(<em>i</em>+1)</sup>)), using your representation of <em>θ</em><sup>(<em>i</em>)</sup>; and (c) How you will modify the update rule given above to perform an update to <em>θ </em>on a new training example (<em>x</em><sup>(<em>i</em>+1)</sup><em>,y</em><sup>(<em>i</em>+1)</sup>); i.e., using the update rule corresponding to the feature mapping <em>φ</em>:</li>

</ul>

<em>θ</em>(<em>i</em>+1) := <em>θ</em>(<em>i</em>) + <em>α</em>[<em>y</em>(<em>i</em>+1) − <em>h<sub>θ</sub></em>(<em>i</em>)(<em>φ</em>(<em>x</em>(<em>i</em>+1)))]<em>φ</em>(<em>x</em>(<em>i</em>+1))

[Note: If you prefer, you are also welcome to do this problem using the convention of labels <em>y </em>∈ {−1<em>,</em>1}, and <em>g</em>(<em>z</em>) = sign(<em>z</em>) = 1 if <em>z </em>≥ 0, −1 otherwise.]

<h1>3.     Spam classification</h1>

In this problem, we will use the naive Bayes algorithm and an SVM to build a spam classifier.

In recent years, spam on electronic newsgroups has been an increasing problem. Here, we’ll build a classifier to distinguish between “real” newsgroup messages, and spam messages. For this experiment, we obtained a set of spam emails, and a set of genuine newsgroup messages.<sup>1 </sup>Using only the subject line and body of each message, we’ll learn to distinguish between the spam and non-spam.

All the files for the problem are in /afs/ir/class/cs229/ps/ps2/. <strong>Note: Please do not circulate this data outside this class. </strong>In order to get the text emails into a form usable by naive Bayes, we’ve already done some preprocessing on the messages. You can look at two sample spam emails in the files spamsampleoriginal*, and their preprocessed forms in the files spamsamplepreprocessed*. The first line in the preprocessed format is just the label and is not part of the message. The preprocessing ensures that only the message body and subject remain in the dataset; email addresses (EMAILADDR), web addresses (HTTPADDR), currency (DOLLAR) and numbers (NUMBER) were also replaced by the special tokens to allow them to be considered properly in the classification process. (In this problem, we’ll going to call the features “tokens” rather than “words,” since some of the features will correspond to special values like EMAILADDR. You don’t have to worry about the distinction.) The files newssampleoriginal and newssamplepreprocessed also give an example of a non-spam mail.

The work to extract feature vectors out of the documents has also been done for you, so you can just load in the design matrices (called document-word matrices in text classification) containing all the data. In a document-word matrix, the <em>i<sup>th </sup></em>row represents the <em>i<sup>th </sup></em>document/email, and the <em>j<sup>th </sup></em>column represents the <em>j<sup>th </sup></em>distinct token. Thus, the (<em>i,j</em>)-entry of this matrix represents the number of occurrences of the <em>j<sup>th </sup></em>token in the <em>i<sup>th </sup></em>document.

For this problem, we’ve chosen as our set of tokens considered (that is, as our vocabulary) only the medium frequency tokens. The intuition is that tokens that occur too often or too rarely do not have much classification value. (Examples tokens that occur very often are words like “the,” “and,” and “of,” which occur in so many emails and are sufficiently content-free that they aren’t worth modeling.) Also, words were stemmed using a standard stemming algorithm; basically, this means that “price,” “prices” and “priced” have all been replaced with “price,” so that they can be treated as the same word. For a list of the tokens used, see the file TOKENSLIST.

Since the document-word matrix is extremely sparse (has lots of zero entries), we have stored it in our own efficient format to save space. You don’t have to worry about this format.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> The file readMatrix.m provides the readMatrix function that reads in the document-word matrix and the correct class labels for the various documents. Code in nb train.m and nb test.m shows how readMatrix should be called. The documentation at the top of these two files will tell you all you need to know about the setup.

<ul>

 <li>Implement a naive Bayes classifier for spam classification, using the multinomial eventmodel and Laplace smoothing.</li>

</ul>

You should use the code outline provided in nbtrain.m to train your parameters, and then use these parameters to classify the test set data by filling in the code in nbtest.m. You may assume that any parameters computed in nbtrain.m are in memory when nb test.m is executed, and do not need to be recomputed (i.e., that nbtest.m is executed immediately after nbtrain.m) <a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>.

Train your parameters using the document-word matrix in MATRIX.TRAIN, and then report the test set error on MATRIX.TEST.

<strong>Remark. </strong>If you implement naive Bayes the straightforward way, you’ll find that the computed <em>p</em>(<em>x</em>|<em>y</em>) = Q<em>i p</em>(<em>x<sub>i</sub></em>|<em>y</em>) often equals zero. This is because <em>p</em>(<em>x</em>|<em>y</em>), which is the product of many numbers less than one, is a very small number. The standard computer representation of real numbers cannot handle numbers that are too small, and instead rounds them off to zero. (This is called “underflow.”) You’ll have to find a way to compute naive Bayes’ predicted class labels without explicitly representing very small numbers such as <em>p</em>(<em>x</em>|<em>y</em>). [Hint: Think about using logarithms.]

<ul>

 <li>Intuitively, some tokens may be particularly indicative of an email being in a particular class. We can try to get an informal sense of how indicative token <em>i </em>is for the SPAM class by looking at:</li>

</ul>

<em> .</em>

Using the parameters fit in part (a), find the 5 tokens that are most indicative of the SPAM class (i.e., have the highest positive value on the measure above). The numbered list of tokens in the file TOKENSLIST should be useful for identifying the words/tokens.

<ul>

 <li>Repeat part (a), but with training sets of size ranging from 50, 100, 200, …, up to 1400, by using the files TRAIN.*. Plot the test error each time (use MATRIX.TEST as the test data) to obtain a learning curve (test set error vs. training set size). You may need to change the call to readMatrix in nbtrain.m to read the correct file each time. Which training-set size gives the best test set error?</li>

 <li>Train an SVM on this dataset using the LIBLINEAR SVM library, available for download from http://www.csie.ntu.edu.tw/˜cjlin/liblinear/. This implements an SVM using a linear kernel. Like the Naive Bayes implementation, an outline for your code is provided in m and svmtest.m.</li>

</ul>

See ps2/README.txt for instructions for downloading and installing LIBLINEAR. Similar to part (c), train an SVM with training set sizes 50, 100, 200, …, 1400, by using the file MATRIX.TRAIN.50 and so on. Plot the test error each time, using MATRIX.TEST as the test data. Use the LIBLINEAR default options when training and testing. You don’t need to try different parameter values.

Running LIBLINEAR in Matlab on Windows or Octave can be buggy, depending on which version of Windows you run. We recommend that you use Matlab on the corn machines (e.g., ssh to corn.stanford.edu). However, there are command line programs you can run (without using MATLAB) which are located in liblinear-1.7/windows for Windows and liblinear-1.7/ for Linux/Unix. If you do it this way, please include the commands that you run from the command line in your solution.

<ul>

 <li>How do naive Bayes and Support Vector Machines compare (in terms of generalizationerror) as a function of the training set size?</li>

</ul>

<h1>4.    [20 points] Properties of VC dimension</h1>

In this problem, we investigate a few properties of the Vapnik-Chervonenkis dimension, mostly relating to how VC(<em>H</em>) increases as the set <em>H </em>increases. For each part of this problem, you should state whether the given statement is true, and justify your answer with either a formal proof or a counter-example.

<ul>

 <li>Let two hypothesis classes <em>H</em><sub>1 </sub>and <em>H</em><sub>2 </sub>satisfy <em>H</em><sub>1 </sub>⊆ <em>H</em><sub>2</sub>. Prove or disprove: VC(<em>H</em><sub>1</sub>) ≤ VC(<em>H</em><sub>2</sub>).</li>

 <li>Let <em>H</em><sub>1 </sub>= <em>H</em><sub>2 </sub>∪{<em>h</em><sub>1</sub><em>,…,h<sub>k</sub></em>}. (I.e., <em>H</em><sub>1 </sub>is the union of <em>H</em><sub>2 </sub>and some set of <em>k </em>additional hypotheses.) Prove or disprove: VC(<em>H</em><sub>1</sub>) ≤ VC(<em>H</em><sub>2</sub>) + <em>k</em>. [Hint: You might want to start by considering the case of <em>k </em>= 1.]</li>

 <li>Let <em>H</em><sub>1 </sub>= <em>H</em><sub>2 </sub>∪ <em>H</em><sub>3</sub>. Prove or disprove: VC(<em>H</em><sub>1</sub>) ≤ VC(<em>H</em><sub>2</sub>) + VC(<em>H</em><sub>3</sub>).</li>

</ul>

<h1>5.     Training and testing on different distributions</h1>

In the discussion in class about learning theory, a key assumption was that we trained and tested our learning algorithms on the same distribution D. In this problem, we’ll investigate one special case of training and testing on different distributions. Specifically, we will consider what happens when the training labels are <em>noisy</em>, but the test labels are not.

Consider a binary classification problem with labels <em>y </em>∈ {0<em>,</em>1}, and let D be a distribution over (<em>x,y</em>), that we’ll think of as the original, “clean” or “uncorrupted” distribution. Define D<em><sub>τ </sub></em>to be a “corrupted” distribution over (<em>x,y</em>) which is the same as D, except that the labels <em>y </em>have some probability 0 ≤ <em>τ &lt; </em>0<em>.</em>5 of being flipped. Thus, to sample from D<em><sub>τ</sub></em>, we would first sample (<em>x,y</em>) from D, and then with probability <em>τ </em>(independently of the observed <em>x </em>and <em>y</em>) replace <em>y </em>with 1 − <em>y</em>. Note that D<sub>0 </sub>= D.

The distribution D<em><sub>τ </sub></em>models a setting in which an unreliable human (or other source) is labeling your training data for you, and on each example he/she has a probability <em>τ </em>of mislabeling it. Even though our training data is corrupted, we are still interested in evaluating our hypotheses with respect to the original, uncorrupted distribution D. We define the generalization error <em>with respect to </em>D<em><sub>τ </sub></em>to be

<em>ε<sub>τ</sub></em>(<em>h</em>) = <em>P</em><sub>(<em>x,y</em></sub>)∼D<em><sub>τ</sub></em>[<em>h</em>(<em>x</em>) 6= <em>y</em>]<em>.</em>

Note that <em>ε</em><sub>0</sub>(<em>h</em>) is the generalization error with respect to the “clean” distribution; it is with respect to <em>ε</em><sub>0 </sub>that we wish to evaluate our hypotheses.

<ul>

 <li>For any hypothesis <em>h</em>, the quantity <em>ε</em><sub>0</sub>(<em>h</em>) can be calculated as a function of <em>ε<sub>τ</sub></em>(<em>h</em>) and</li>

</ul>

<em>τ</em>. Write down a formula for <em>ε</em><sub>0</sub>(<em>h</em>) in terms of <em>ε<sub>τ</sub></em>(<em>h</em>) and <em>τ</em>, and justify your answer.

<ul>

 <li>Let |<em>H</em>| be finite, and suppose our training set <em>S </em>= {(<em>x</em><sup>(<em>i</em>)</sup><em>,y</em><sup>(<em>i</em>)</sup>);<em>i </em>= 1<em>,…,m</em>} is obtained by drawing <em>m </em>examples IID from the corrupted distribution D<em><sub>τ</sub></em>. Suppose we pick <em>h </em>∈ <em>H </em>using empirical risk minimization: <em><sup>h</sup></em>ˆ = argmin<em><sub>h</sub></em>∈<em><sub>H </sub>ε</em>ˆ<em><sub>S</sub></em>(<em>h</em>). Also, let <em>h</em><sup>∗ </sup>= argmin<em><sub>h</sub></em>∈<em><sub>H </sub>ε</em><sub>0</sub>(<em>h</em>).</li>

</ul>

Let any <em>δ,γ &gt; </em>0 be given. Prove that for

<em>ε</em><sub>0</sub>(<em>h</em><sup>ˆ</sup>) ≤ <em>ε</em><sub>0</sub>(<em>h</em><sup>∗</sup>) + 2<em>γ</em>

to hold with probability 1 − <em>δ</em>, it suffices that

<em>.</em>

<strong>Remark. </strong>This result suggests that, roughly, <em>m </em>examples that have been corrupted at noise level <em>τ </em>are worth about as much as (1 − 2<em>τ</em>)<sup>2</sup><em>m </em>uncorrupted training examples. This is a useful rule-of-thumb to know if you ever need to decide whether/how much to pay for a more reliable source of training data. (If you’ve taken a class in information theory, you may also have heard that (1−H(<em>τ</em>))<em>m </em>is a good estimate of the information in the <em>m </em>corrupted examples, where H(<em>τ</em>) = −(<em>τ </em>log<sub>2 </sub><em>τ </em>+ (1 − <em>τ</em>)log<sub>2</sub>(1 − <em>τ</em>)) is the “binary entropy” function. And indeed, the functions (1−2<em>τ</em>)<sup>2 </sup>and 1−H(<em>τ</em>) are quite close to each other.)




CS229 Problem Set #2                                                                                                                                            6

(c) Comment <strong>briefly </strong>on what happens as <em>τ </em>approaches 0<em>.</em>5.

<a href="#_ftnref1" name="_ftn1">[1]</a> Thanks to Christian Shelton for providing the spam email. The non-spam messages are from the 20 newsgroups data at http://www-2.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html .

<a href="#_ftnref2" name="_ftn2">[2]</a> Unless you’re not using Matlab/Octave, in which case feel free to ask us about it.

<a href="#_ftnref3" name="_ftn3">[3]</a> Matlab note: If a .m file doesn’t begin with a function declaration, the file is a script. Variables in a script are put into the global namespace, unlike with functions.