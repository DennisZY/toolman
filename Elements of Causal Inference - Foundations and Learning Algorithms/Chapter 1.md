# 1 Statistical and Causal Models

> 统计和因果模型

Using statistical learning, we try to infer properties of the dependence among ran-dom variables from observational data.  

> 使用统计学习，我们尝试从观测数据中推断随机变量之间的依存关系。

For instance, based on a joint sample ofobservations of two random variables, we might build a predictor that, given newvalues of only one of them, will provide a good estimate of the other one.  

> 例如，基于对两个随机变量的联合观察样本，我们可以建立一个预测变量，只要给定其中一个的新值，就可以很好地估计另一个变量。

The theory underlying such predictions is well developed, and — although it applies to simple settings — already provides profound insights into learning from data. 

> 这样的预测所依据的理论已经十分成熟，即便它只通过简单的设置，就可以从数据中学习到了深刻的理解。

For two reasons, we will describe some of these insights in the present chapter.

> ？出于两个原因，我们将在本章描述其中的一些见解。

First, this  will  help  us  appreciate  how  much  harder  the  problems  of causal inference are, where the underlying model is no longer a fixed joint distribution of random variables, but a structure that implies multiple such distributions. 

> ？首先，这会帮助我们了解到因果推断的问题到底有多难，**不会翻**

Second, although finite sample results for causal estimation are scarce, it is important to keep in mind that the basic statistical estimation problems do not go away when moving to the more complex causal setting, even if they seem small compared to the causal problems that do not appear in purely statistical learning. 

> 其次，即便是xxx有限样本结果很少，但是更应该记住的是，即使运用更加复杂的因果设置，基础统计分析的问题也不会消失，即便它看起来有细小的因果问题不会出现像纯粹的统计学习中。

Building on the preceding groundwork, the chapter also provides a gentle introduction to the basic notions of causality, using two examples, one of which is well known from machine learning.

> 在前面的基础上，本章也会通过两个例子对因果关系基本概念进行简要的介绍，其中一个是众所周知的机器学习。

> 这样的预测所依据的理论已经得到了很好的发展，尽管它适用于简单的环境，但已经为从数据中学习提供了深刻的见解。出于两个原因，我们将在本章中描述其中一些见解。首先，这将帮助我们理解因果关系的困难程度，因为基础模型不再是随机变量的固定联合分布，而是表示多个此类分布的结构。其次，尽管用于因果估计的有限样本结果很少，但重要的是要记住，即使移至更复杂的因果环境，基本统计估计问题也不会消失，即使与没有因果问题的因果问题相比，它们似乎很小出现在纯粹的统计学习中。在前面的基础上，本章还使用两个示例对因果关系的基本概念进行了简要介绍，其中一个是机器学习众所周知的。

## 1.1 Probability Theory and Statistics

Probability theory and statistics are based on the model of a random experiment or probability space$(\Omega,\mathcal{F},P)$.

> 概率论和统计学都是基于随机化实验和概率空间的模型

Here, $\Omega$ is a set (containing all possible outcomes),F is a collection of events $A\subseteq \Omega$, and $P$ is a measure assigning a probability to each event.  

> 在这里，$\Omega$是一个集合（包含所有可能的结果），$\mathcal{F}$是事件在$\Omega$的一个子集 $A\subseteq \Omega$，$P$是为每一个时间分配概率的度量。

Probability theory allows us to reason about the outcomes of random experiments, given the preceding mathematical structure.  

> 通过给定前面的数学**结构**，概率论让我们能推理随机实验的结果。

Statistical learning, on the other hand, essentially deals with the inverse problem:  We are given the out-comes of experiments, and from this we want to infer properties of the underlying mathematical structure.

> ？从另一方面来说，统计学习是用来解决反问题的：我们得到实验的结果，并且推到出基础数学结构。

For instance, suppose that we have observed data
$$
(x1,y1),...,(xn,yn),\label{data:1}
$$

where $x_i \in \mathcal{X}$ are **inputs**(sometimes called covariates or cases) and $y_i\in \mathcal Y$are **outputs**(sometimes called targets or labels).  

> 例如，假设我们有观察数据，$x_i$为输入（有时也称为协变量和个案），$y_i$是输出（有时也称为目标和标签）。

We  may  now  assume  that  each $(x_i,y_i),i=1,...,n$, has been generated independently by the same unknown random  experiment.

> 我们现在假设这些$(x_i,y_i)$是由相同的未知随机实验生成的。

More  precisely,  such  a  model  assumes  that  the  observations$(x_1,y_1),...,(x_n,y_n)$ are realizations of random variables$(X1,Y1),...,(Xn,Yn)$that are **i.i.d. (independent and identically distributed)** with joint distribution $P_{X,Y}$. Here, $X$ and $Y$ are random variables taking values in metric spaces$\mathcal X$ and $\mathcal Y$ .$^1$有脚注 page2 pdf19

> 更准确地说，这样的模型假设观测值$(x_1,y_1),...,(x_n,y_n)$ 是随机变量$(X1,Y1),...,(Xn,Yn)$的实现是**i.i.d (independent and identically distributed)**（独立且均匀分布）联合分布$P_{X,Y}$。此处 $X$ 和 $Y$ 是在度量空间 $\mathcal X$ 和 $\mathcal Y$ 中取值的随机变量.

Almost all of statistics and machine learning builds on i.i.d. data.

>几乎所有统计和机器学习都基于i.i.d.数据

In practice, the i.i.d.assumption can be violated in various ways, for instance if distributions shift or interventions in a system occur. 

> 在实践上，可以以多种方式违反i.i.d.假设，例如，如果分布发生偏移或系统中发生干预。

As we shall see later, some of these are intricately linked to causality.

> 稍后我们将看到，其中一些与因果关系错综复杂。

We may now be interested in certain properties of$P_{X,Y}$, such as:

> 我们可能会对$P_{X,Y}$的某些属性感兴趣，例如：

(i)  the  expectation  of  the  output  given  the  input, $f(x) =\mathbb E[Y|X=x]$,  called regression, where often $\mathcal Y= \mathbb R$,

> 给定输入的输出期望$f(x) =\mathbb E[Y|X=x]$，称为回归，其中通常$\mathcal Y= \mathbb R$，

(ii)  a binary classifier assigning each x to the class that is more likely, $f(x) =argmax_{y∈\mathcal Y}P(Y=y|X=x)$, where$Y=\{±1\}$,

> 一个二元分类器，将每个$x$分配给更可能的类，$f(x) =argmax_{y∈\mathcal Y}P(Y=y|X=x)$，其中Y = {±1}，

(iii)  the density $p_{X,Y}$ of $P_{X,Y}$ (assuming it exists).

> 密度 $p_{X,Y}$of$P_{X,Y}$（假设存在）。

In practice, we seek to estimate these properties from finite data sets, that is, basedon the sample (1.1), or equivalently an empirical distribution$P^n_{X,Y}$that puts a pointmass of equal weight on each observation.

> 在实践中，我们试图从有限数据集中估算这些属性，即基于样本 $\ref{data:1}$或等效的经验分布$P^n_{X,Y}$，其点为每个观察值的权重相等。

This constitutes an inverse problem: We want to estimate a property of an object we  cannot  observe  (the  underlying  distribution),  based  on  observations  that  are obtained by applying an operation (in the present case: sampling from the unknown distribution) to the underlying  object.

> 这就构成了一个相反的问题：我们想要基于对基础对象执行操作（在当前情况下：从未知分布中采样）所获得的观察结果，来估计无法观察的对象的属性（基础分布）。

> 概率论和统计学基于随机实验或概率空间（Ω，F，P）的模型。这里，Ω是一个集合（包含所有可能的结果），Fis是事件的集合A⊆Ω，Pis是为每个事件分配概率的度量。给定前面的数学结构，概率论使我们能够对随机实验的结果进行推理。统计学习，关于
>  另一方面，本质上是解决反问题的：我们得到了实验的结果，因此我们要推断出基础数学结构的性质。例如，假设我们观察到数据（x1，y1），...，（xn，yn），（1.1）其中xi∈Xareinputs（有时称为协变量或个案）和yi∈Yareoutputs（有时称为targetsorlabels）。现在我们可以假设，每个（xi，yi），i = 1，...，n是由相同的未知随机实验独立生成的。更准确地说，这样的模型假设观测值（x1，y1），...，（xn，yn）是随机变量（X1，Y1），...，（Xn，Yn）的实现。 （独立且相同分布）联合分布PX，Y。此处X和Y是在度量空间X和Y中取值的随机变量.1Al-几乎所有统计和机器学习都基于i.i.d.数据。在实践中，可以以多种方式违反i.i.d.假设，例如，如果分布发生变化或系统中发生干预。稍后我们将看到，其中一些与因果关系错综复杂。我们稍后可能会对PX，Y的某些属性感兴趣，例如：（i）给定输入的输出期望f（x）= E [Y | X = x]，称为回归，其中通常Y = R，（ii）一个二元分类器，将每个x分配给更可能的类，f（x）=argmaxy∈YP（Y = y | X = x），其中Y = {±1}， （iii）密度pX，YofPX，Y（假设存在）。在实践中，我们试图从有限数据集中估算这些属性，即基于样本（1.1）或等效的经验分布PnX，Y，其点为每个观察值的权重相等。这构成了一个反问题：我们要基于对操作进行操作（在当前情况下：从未知分布中采样）获得的观察值来估计无法观察到的对象的属性（基础分布）。基础对象。

## 1.2 Learning Theory

Now suppose that just like we can obtain f from PX,Y, we use the empirical distribution to infer empirical estimatesfn. This turns out to be anill-posed problem[e.g., Vapnik, 1998], since for any values ofxthat we have not seen in the sample(x1,y1),...,(xn,yn), the conditional expectation is undefined.  We may, however,define the functionfon the observed sample and extend it according to any fixedrule (e.g., settingfto+1 outside the sample or by choosing a continuous piecewiselinearf).  But for any such choice, small changes in the input, that is, in the em-pirical distribution, can lead to large changes in the output.  

No matter how manyobservations we have, the empirical distribution will usually not perfectly approx-imate the true distribution,  and small errors in this approximation can then lead to large errors in the estimates.  

> 不管我们有多少个观测值，经验分布通常都不能完美地逼近真实分布，并且这种近似中的小误差会导致估计中的大误差。

This implies that without additional assumptions about the class of functions from which we choose our empirical estimates fn, we cannot guarantee that the estimates will approximate the optimal quantities f in asuitable sense.  

> 这意味着，如果没有关于我们从中选择经验估计值fn的函数类别的其他假设，就无法在适当的意义上保证估计值将近似于最佳数量。

In statistical learning theory, these assumptions are formalized in terms of capacity measures.  If we work with a function class that is so rich that it can fit most conceivable data sets, then it is not surprising if we can fit the data at hand. If, however, the function class is a priori(先验) restricted to have small capacity, then there are only a few data sets (out of the space of all possible data sets) that we can explain using a function from that class. If it turns out that nevertheless we can explain the data at hand, then we have reason to believe that we have found aregularity underlying the data.  In that case, we can give probabilistic guarantees for the solution’s accuracy on future data sampled from the same distribution $P_{X,Y}$.

> 现在假设就像我们可以从PX，Y获得f一样，我们使用经验分布来推断经验估计f n。事实证明这是一个病态的问题[例如，Vapnik，1998年]，因为对于我们在样本（x1，y1），...，（xn，yn）中未看到的任何x值，条件期望是不确定的。但是，我们可以定义所观察样本的函数，并根据任何固定规则（例如，将fto设置为样本外部的fto + 1或通过选择连续的分段线性函数f）对其进行扩展。但是对于任何这样的选择，输入（即经验分布）中的微小变化都可能导致输出中的较大变化。不管我们有多少个观测值，经验分布通常都不能完美地逼近真实分布，并且这种近似中的小误差会导致估计中的大误差。这意味着，如果没有关于我们从中选择经验估计值fn的函数类别的其他假设，就无法在适当的意义上保证估计值将近似于最佳数量。在统计学习理论中，这些假设是容量度量的形式化形式。如果我们使用的函数类非常丰富，可以适合大多数可能的数据集，那么我们适合手头的数据就不足为奇了。但是，如果函数类被先验地限制为具有小容量，那么只有少数数据集（在所有可能的数据集的空间之外），我们可以使用该类中的函数进行解释。如果事实证明我们仍然可以解释手头的数据，那么我们就有理由相信我们已经发现了数据背后的细微之处。在这种情况下，我们可以为从同一分布PX，Y采样的未来数据的解决方案的准确性提供概率保证。

Another way to think of this is that our function class has incorporateda priori knowledge(such as smoothness of functions) consistent with the regularity underlying the observed data. Such knowledge can be incorporated in various ways, and different approaches to machine learning differ in how they handle the issue. In Bayesian approaches, we specify prior distributions over function classes and noisemodels. In regularization theory, we construct suitable regularizers and incorporate them into optimization problems to bias our solutions.

> 另一种思考的方式是我们的函数类已经合并了一个先验知识（例如函数的平滑度），该知识与作为观察数据基础的规则性一致。 可以以各种方式合并这些知识，并且机器学习的不同方法在处理问题方面也有所不同。 在贝叶斯方法中，我们指定了函数类和噪声模型的先验分布。 在正则化理论中，我们构造合适的正则化器并将其合并到优化问题中以使我们的解决方案产生偏差。

The complexity of statistical learning arises primarily from the fact that we are trying to solve an inverse problem based on empirical data — if we were given the full probabilistic model, then all these problems go away.  When we discuss causal models, we will see that in a sense, the causal learning problem is harder in  that it is ill-posed on two levels. In addition to the statistical  ill-posed-ness, which is essentially because a finite sample of arbitrary size will never contain all information about the underlying distribution, there is an ill-posed-ness due to the fact that even complete knowledge of an observational distribution usually doesnot determine the underlying causal model.

> 统计学习的复杂性主要来自以下事实：我们试图基于经验数据来解决逆问题-如果我们获得了完整的概率模型，那么所有这些问题都将消失。 在讨论因果模型时，从某种意义上说，因果学习问题更难，因为它在两个层面上是不适当的。 除了统计上的不适性（本质上是因为任意大小的有限样本将永远不会包含有关基础分布的所有信息）外，还存在不适性的原因，因为即使对观测资料的完全了解 分布通常不能确定潜在的因果模型。

Let us look at the statistical learning problem in more detail,  focusing on the case of binary pattern recognition or classification [e.g., Vapnik, 1998], where $Y=\{±1\}$.  We seek to learn$f:\mathcal X →\mathcal Y $ based on observations, generated i.i.d. from an unknown$P_{X,Y}$ Our goal is to minimize the expected error or risk

> 让我们更详细地研究统计学习问题，着眼于二进制模式识别或分类的情况[e.g.，Vapnik，1998]，其中$Y=\{±1\}$。 我们试图根据观察结果学习，$f:\mathcal X →\mathcal Y $ 即i.i.d. 来自未知的$ P_ {X，Y} $。我们的目标是最大程度地减少预期的错误或风险。

$$
R[f]=\int \frac{1}{2}|f(x)-y|dP_{X,Y}(x,y)
$$

over some class of functions F.  Note that this is an integral with respect to the measure PX,Y; however, if PX,Y allows for a density p(x,y) with respect to Lebesgue measure, the integral reduces to∫12|f(x)−y|p(x,y)dxdy.

> 注意，这对于度量PX，Y是不可或缺的。 但是，如果相对于Lebesgue测度，PX，Y允许密度p（x，y），则积分减小为∫12| f（x）-y | p（x，y）dxdy。

Since PX,Y is unknown, we cannot compute (1.2), let alone minimize it. Instead,we appeal to an induction principle, such as empirical risk minimization.  We return the function minimizing the training error or empirical risk

> 由于PX，Y是未知的，因此我们无法计算（1.2），更不用说将其最小化了。 相反，我们呼吁采用归纳原则，例如经验风险最小化。 我们返回使训练误差或经验风险最小化的函数

$$
R^n_{emp}[f]=\frac{1}{n}\sum^n_{i=1}\frac{1}{2}|f(x_i)-y_i|
$$

overf∈F.   From the asymptotic point of view,  it is important to ask whether such  a  procedure  is consistent,  which  essentially  means  that  it  produces  a  sequence  of  functions  whose  risk  converges  towards  the  minimal  possible  within the given function class F(in probability) as n tends to infinity. In Appendix A.3,we show that this can only be the case if the function class is “small.” The Vapnik-Chervonenkis (VC) dimension [Vapnik, 1998] is one possibility of measuring the capacity or size of a function class. It also allows us to derive finite sample guarantees, stating that with high probability, the risk (1.2) is not larger than the empirical risk plus a term that grows with the size of the function class F.

> 从渐近的观点来看，重要的是要问这样一个过程是否一致，这实质上意味着它会生成一系列函数，当n趋于无限时，该函数的风险在给定函数类F（概率）内收敛到最小可能。 在附录A.3中，我们表明只有在函数类为“ small”的情况下才是这种情况。 Vapnik-Chervonenkis（VC）的维数[Vapnik，1998]是衡量功能类的容量或大小的一种可能性。 它还允许我们得出有限的样本保证，指出风险（1.2）的概率不大于经验风险加上随函数类F的大小而增加的项。

Such a theory does not contradict the existing results on universal consistency, which refers to convergence of a learning algorithm to the lowest achievable risk with any function.  There are learning algorithms that are universally consistent, for instance nearest neighbor classifiers and Support Vector Machines [Devroyeet al., 1996, Vapnik, 1998, Sch ̈olkopf and Smola, 2002, Steinwart and Christmann,2008]. While universal consistency essentially tells us everything can be learned in the limit of infinite data, it does not imply that every problem is learnable well from finite data, due to the phenomenon of slow rates. For any learning algorithm, there exist  problems  for  which  the  learning  rates  are  arbitrarily  slow  [Devroye  et  al.,1996].  It does tell us, however, that if we fix the distribution, and gather enough data, then we can get arbitrarily close to the lowest risk eventually.

> 这样的理论与普遍一致性的现有结果并不矛盾，普遍一致性是指将学习算法收敛到具有任何功能的最低可实现风险。 有一些学习算法是普遍一致的，例如最近邻分类器和支持向量机[Devroyeet等，1996； Vapnik，1998； Sch olkopf和Smola，2002； Steinwart和Christmann，2008]。 尽管通用一致性从本质上告诉我们可以在无限数据的限制中学习所有内容，但这并不意味着由于速率降低的现象，可以从有限数据中很好地学习每个问题。 对于任何一种学习算法，都存在学习速率任意变慢的问题[Devroye et al。，1996]。 但是，它确实告诉我们，如果我们固定分布并收集足够的数据，则最终我们可以任意接近最低风险。

In practice, recent successes of machine learning systems seem to suggest that we are indeed sometimes already in this asymptotic regime, often with spectacular results.  A lot of thought has gone into designing the most data-efficient methods to obtain the best possible results from a given data set, and a lot of effort goes into building large data sets that enable us to train these methods.  However, in all these settings, it is crucial that the underlying distribution does not differ between training and testing, be it by interventions or other changes.  As we shall argue in this book, describing the underlying regularity as a probability distribution, without additional  structure,  does  not  provide  us  with  the  right  means  to  describe  what might change.

> 在实践中，机器学习系统的最新成功似乎表明，我们确实有时已经处于这种渐近状态中，并且通常会产生惊人的结果。 设计最有效的数据方法以从给定的数据集中获得最佳结果的方法已经引起了很多思考，并且花费大量精力来构建大型数据集以使我们能够训练这些方法。 但是，在所有这些情况下，至关重要的是，无论是通过干预还是其他更改，基本分布在培训和测试之间都不得有所不同。 正如我们将在本书中论述的那样，将基本规则性描述为概率分布而没有其他结构，并不能为我们提供描述可能发生变化的正确方法。

## 1.3    Causal Modeling and Learning

Causal  modeling  starts  from  another,  arguably  more  fundamental,  structure.   A causal structure entails a probability model, but it contains additional information not contained in the latter (see the examples in Section 1.4).Causal reasoning,according to the terminology used in this book,  denotes the process of drawing conclusions from a causal model, similar to the way probability theory allows us to reason about the outcomes of random experiments. However, since causal models contain more information than probabilistic ones do, causal reasoning is more powerful than probabilistic reasoning, because causal reasoning allows us to analyze the effect of interventions or distribution changes.

> 因果模型从另一个可以说是更基本的结构开始。 因果结构需要概率模型，但是它包含后者中未包含的其他信息（请参阅第1.4节中的示例）。根据本书所用的术语，因果推理表示从因果模型得出结论的过程， 类似于概率论使我们能够推断随机实验结果的方式。 但是，由于因果模型比概率模型包含更多的信息，因果推理比概率推理更强大，因为因果推理使我们能够分析干预措施或分布变化的影响。

Just like statistical learning denotes the inverse problem to probability theory, we can think about how to infer causal structures from its empirical implications. The empirical implications can be purely observational, but they can also include data under interventions (e.g., randomized trials) or distribution changes.  Researchers use  various  terms  to  refer  to  these  problems,  including structure  learning and causal discovery.  We refer to the closely related question of which parts of the causal structure can in principle be inferred from the joint distribution as structure identifiability. Unlike the standard problems of statistical learning described in Section 1.2, even full knowledge of P does not make the solution trivial, and we need additional assumptions (see Chapters 2, 4, and 7).  This difficulty should not distract us from the fact, however, that the ill-posed-ness of the usual statistical problems is still there (and thus it is important to worry about the capacity of function classes also in causality, such as by using additive noise models — see Section 4.1.4 below), only confounded by an additional difficulty arising from the fact that we are trying to estimate a richer structure than just a probabilistic one.We will refer to this overall problem as causal learning.  Figure 1.1 summarizes the relationships between the preceding problems and models.

> 就像统计学习表示概率论的逆问题一样，我们可以考虑如何从其经验含义中推断因果结构。经验意义可以纯粹是观察性的，但也可以包括干预措施（例如，随机试验）或分布变化下的数据。研究人员使用各种术语来指代这些问题，包括结构学习和因果发现。我们将与之密切相关的问题称为因果结构的哪些部分，原则上可以从联合分布中推断出因果结构的哪些部分。与第1.2节中描述的统计学习的标准问题不同，即使对P的全面了解也不能使解决方案变得微不足道，因此我们需要其他假设（请参阅第2、4和7章）。这个困难不应使我们分心，但通常的统计问题仍然存在问题（因此，担心函数类在因果关系中的能力也很重要，例如使用加法运算，这一点很重要。噪声模型-参见下面的4.1.4节），但又因我们试图估算比概率模型更丰富的结构而引起的额外困难，使我们感到困惑。我们将整个问题称为因果学习。图1.1总结了上述问题和模型之间的关系。 

To learn causal structures from observational distributions, we need to understand how causal models and statistical models relate to each other.  We will come back to this issue in Chapters 4 and 7 but provide an example now. A well-known topos holds that correlation does not imply causation; in other words, statistical properties alone do not determine causal structures.  It is less well known that one may postulate that while we cannot infer a concrete causal structure, we may at least infer the existence of causal links from statistical dependences. This was first understood by Reichenbach [1956]; we now formulate his insight (see also Figure 1.2).3

> 要从观测分布中了解因果结构，我们需要了解因果模型与统计模型之间的关系。 我们将在第4章和第7章中回到这个问题，但现在提供一个示例。 众所周知的主题认为关联并不意味着因果关系。 换句话说，仅统计属性并不能确定因果关系。 鲜为人知的是，人们可能会假设，虽然我们无法推断出具体的因果结构，但我们至少可以从统计依赖性中推断出因果联系的存在。 这是Reichenbach [1956]最初了解的； 现在我们来阐述他的见解（另请参见图1.2）.3

**Principle  1.1  (Reichenbach’s  common  cause  principle)**If  two  random  variables X and Y are statistically dependent (X6⊥⊥Y ), then there exists a third variable Z that causally influences  both.  (As a special case, Z may coincide with either X or Y .)  Furthermore, this variable Z screens X and Y  from each other in the sense that given Z, they become independent, X⊥⊥Y|Z.

> 原理1.1（赖兴巴赫的共因原理）如果两个随机变量X和Y在统计上是相关的（X6⊥⊥Y），那么将存在第三个因变量Z对其产生影响。 （在特殊情况下，Z可以与X或Y一致。）此外，在给定Z的情况下，变量Z相互独立，使X和Y相互屏蔽。

In practice, dependences may also arise for a reason different from the ones mentioned in the common cause principle, for instance:  (1) The random variables we observe are conditioned on others (often implicitly by a selection bias).  We shall return to this issue;  see Remark 6.29.  (2) The random variables only appear to be dependent.  For example, they may be the result of a search procedure over a large number of pairs of random variables that was run without a multiple testing correction. In this case, inferring a dependence between the variables does not satisfy the desired type I error control; see Appendix A.2. (3) Similarly, both random variables may inherit a time dependence and follow a simple physical law, such as exponential growth.  The variables then look as if they depend on each other, but because the i.i.d. assumption is violated, there is no justification of applying a standard independence test.  In particular, arguments (2) and (3) should be kept in mind when reporting “spurious correlations” between random variables, as it is done on many popular websites.

> 在实践中，依赖关系也可能由于与常见原因原理中提到的原因不同而产生，例如：（1）我们观察到的随机变量以其他变量为条件（通常隐含选择偏见）。我们将回到这个问题；见备注6.29。 （2）随机变量似乎仅是因变量。例如，它们可能是针对大量随机变量的搜索过程的结果，这些随机变量是在没有进行多次测试校正的情况下运行的。在这种情况下，推断变量之间的依赖关系不能满足所需的I类错误控制；见附录A.2。 （3）同样，两个随机变量都可以继承时间依赖性，并遵循简单的物理定律，例如指数增长。然后，变量看起来好像它们相互依赖，但这是因为i.i.d.如果违反了假设，则没有理由进行标准独立性测试。特别是，在报告随机变量之间的“虚假相关性”时，应牢记论点（2）和（3），就像在许多流行的网站上所做的那样。

## 1.4 Two Examples

### 1.4.1 Pattern Recognition

As  the  first  example,  we  consider optical  character  recognition,  a  well-studied problem in machine learning.   This is not a run-of-the-mill example of a causal structure, but it may be instructive for readers familiar with machine learning. We describe two causal models giving rise to a dependence between two random variables, which we will assume to be handwritten digits X and class labels Y. The two models will lead to the same statistical structure, using distinct underlying causal structures.

> 作为第一个示例，我们考虑光学字符识别，这是机器学习中经过充分研究的问题。 这不是因果结构的常规示例，但对于熟悉机器学习的读者可能有启发性。我们描述了两个因果模型，它们引起两个随机变量之间的依赖关系，我们将假设它们是手写数字X和类别标签Y。这两个模型将使用不同的基础因果结构得出相同的统计结构。

Model (i) assumes we generate each pair of observations by providing a sequence of class labels y to a human writer, with the instruction to always produce a corresponding handwritten digit image x.  We assume that the writer tries to do a good job, but there may be noise in perceiving the class label and executing the motor program to draw the image. We can model this process by writing the image X as a function (or mechanism) f of the class label Y(modeled as a random variable) andsome independent noise NX (see Figure 1.3, left). We can then compute PX, Y from PY, PNX, and f.  This is referred to as the observational  distribution, where the word “observational” refers to the fact that we are passively observing the system without intervening. X and Y will be dependent random variables, and we will beable to learn the mapping from x to y from observations and predict the correct label y from an image x better than chance.

> 模型（i）假设我们通过向人类作家提供一系列类别标签y来生成每一对观察值，并始终生成相应的手写数字图像x的指令。 我们假设作者尝试做得很好，但是在感知类标签并执行马达程序以绘制图像时可能会有些杂音。 我们可以通过将图像X写入类标签Y（建模为随机变量）和某些独立噪声NX的函数（或机制）f（参见左图1.3）来对该过程进行建模。 然后，我们可以根据PY，PNX和f计算PX，Y。 这被称为观测分布，其中“观测”一词指的是我们在不干预的情况下被动观测系统的事实。 X和Y将是因变量，因此我们将能够从观察中学习从x到y的映射，并比偶然性更好地从图像x预测正确的标签y。

There are two possible interventions in this causal structure, which lead to intervention distributions. If we intervene on the resulting image X(by manipulating it, or exchanging it for another image after it has been produced), then this has no effect on the class labels that were provided to the writer and recorded in the dataset. Formally, changing X has no effect on Y since Y:=NY. Intervening on Y, on the other hand, amounts to changing the class labels provided to the writer.  This will obviously have a strong effect on the produced images. Formally, changing Y has an effect on X since X:=f(Y,NX).  This directionality is visible in the arrowin the figure, and we think of this arrow as representing direct causation.

> 在这种因果结构中有两种可能的干预措施，导致干预措施的分布。 如果我们干预生成的图像X（通过对其进行操作，或在生成图像后将其交换为另一图像），则这对提供给编写者并记录在数据集中的类标签没有影响。 正式地，因为Y：= NY，更改X对Y没有影响。 另一方面，干预Y等于更改提供给编写者的类标签。 显然，这将对产生的图像产生强烈影响。 形式上，更改X会影响X，因为X：= f（Y，NX）。 该方向性在图中的箭头中可见，我们认为该箭头表示直接因果关系。

In alternative model (ii), we assume that we do not provide class labels to the writer. Rather, the writer is asked to decide himself or herself which digits to write, and to record the class labels alongside.  In this case, both the image X and the recorded class label Y are functions of the writer’s intention (call it Z and think of it as a random variable).  For generality, we assume that not only the process generating the image is noisy but also the one recording the class label, again within dependent noise terms (see Figure 1.3, right). Note that if the functions and noise terms are chosen suitably, we can ensure that this model entails an observational distribution PX,Y that is identical to the one entailed by model (i).

> 在替代模型（ii）中，我们假定不向编写者提供类标签。 相反，要求作者自己决定要写哪些数字，并在旁边记录班级标签。 在这种情况下，图像X和记录的类别标签Y都是作者意图的函数（将其称为Z并将其视为随机变量）。 为了通用起见，我们假设不仅生成图像的过程有噪声，而且记录类标签的过程也是有噪声的（同样参见图1.3，右）。 注意，如果适当地选择了函数和噪声项，我们可以确保该模型的观测分布PX，Y与模型（i）的观测分布相同。

Let us now discuss possible interventions in model (ii).  If we intervene on the image X, then things are as we just discussed and the class label Y is not affected. However, if we intervene on the class label Y(i.e., we change what the writer has recorded as the class label), then unlike before this will not affect the image.

> 现在让我们讨论模型（ii）中的可能干预措施。 如果我们对图像X进行干预，则情况就如我们刚才讨论的那样，并且类别标签Y不会受到影响。 但是，如果我们干预类标签Y（即，我们更改编写者记录为类标签的内容），则与之前不同，这不会影响图像。

In summary, without restricting the class of involved functions and distributions, the causal models described in (i) and (ii) induce the same observational distribution over X and Y, but different intervention distributions.  This difference is not visible in a purely probabilistic description (where everything derives from PX,Y). However, we were able to discuss it by incorporating structural knowledge about how PX,Y comes about, in particular graph structure, functions, and noise terms.

> 总之，在不限制所涉及功能和分布的类别的情况下，（i）和（ii）中描述的因果模型在X和Y上产生相同的观察分布，但是在干预分布上却不同。 在纯粹的概率描述中（所有事物均来自PX，Y），这种差异是不可见的。 但是，我们能够通过结合有关PX，Y的结构知识来讨论它，特别是图形结构，函数和噪声项。

Models (i) and (ii) are examples of structural causal models (SCMs), sometimes  referred  to  as structural  equation  models[e.g.,  Aldrich,  1989,  Hoover,2008, Pearl, 2009, Pearl et al., 2016]. In an SCM, all dependences are generated by functions that compute variables from other variables.  Crucially, these functions are to be read as assignments, that is, as functions as in computer science rather than as mathematical equations.  We usually think of them as modeling physical mechanisms.  An SCM entails a joint distribution over all observables.  We have seen that the same distribution can be generated by different SCMs, and thus information about the effect of interventions (and,  as we shall see in Section 6.4,information about counterfactuals) may be lost when we make the transition from an SCM to the corresponding probability model.  In this book, we take SCMs as our starting point and try to develop everything from there.

> 模型（i）和（ii）是结构因果模型（SCM）的示例，有时也称为结构方程模型[例如，Aldrich，1989； Hoover，2008； Pearl，2009； Pearl等，2016]。 在SCM中，所有依赖关系均由从其他变量计算变量的函数生成。 至关重要的是，这些函数应作为赋值来读取，即作为计算机科学中的函数而不是数学方程式。 我们通常认为它们是对物理机制的建模。 SCM要求在所有可观测物上进行联合分配。 我们已经看到，不同的SCM可能会产生相同的分布，因此，当我们从SCM向相应的概率模型过渡时，有关干预效果的信息（以及我们将在6.4节中看到的有关反事实的信息）可能会丢失。在本书中，我们将SCM作为起点，并尝试从那里开始开发一切。

We conclude with two points connected to our example:

> 我们以与示例相关的两点作为结论：

First, Figure 1.3 nicely illustrates Reichenbach’s common cause principle.  The dependence  between X and Y admits  several  causal  explanations,  and X and Y become independent if we condition on Z in the right-hand figure: The image and the label share no information that is not contained in the intention.

> 首先，图1.3很好地说明了赖兴巴赫的共因原理。 X和Y之间的依存关系有几种因果关系，如果我们以右侧图中的Z为条件，则X和Y将变得独立：图像和标签不共享意图中未包含的信息。

Second, it is sometimes said that causality can only be discussed when taking into account the notion of time.  Indeed,  time does play a role in the preceding example, for instance by ruling out that an intervention onXwill affect the class label.  However,  this is perfectly fine,  and indeed it is quite common that a statistical data set is generated by a process taking place in time.   For instance,  in model (i), the underlying reason for the statistical dependence between X and Y is a dynamical process.  The writer reads the label and plans a movement, entailing complicated processes in the brain, and finally executes the movement using muscles and a pen.   This process is only partly understood,  but it is a physical, dynamical process taking place in time whose end result leads to a non-trivial joint distribution of X and Y. When we perform statistical learning, we only care about the end result. Thus, not only causal structures, but also purely probabilistic structures may arise through processes taking place in time — indeed, one could holdthat this is ultimately the only way they can come about.  However, in both cases,it is often instructive to disregard time.  In statistics, time is often not necessary to discuss concepts such as statistical dependence. In causal models, time is often not necessary to discuss the effect of interventions.  But both levels of description can be thought of as abstractions of an underlying more accurate physical modelthat describes reality more fully than either;  see Table 1.1.  Moreover,  note that variables in a model may not necessarily refer to well-defined time instances.  If, for instance, a psychologist investigates the statistical or causal relation betweenthe motivation and the performance of students,  both variables cannot easily be assigned to specific time instances.  Measurements that refer to well-defined timeinstances are rather typical for “hard” sciences like physics and chemistry.

> 其次，有时会说因果关系只能在考虑时间概念的情况下进行讨论。实际上，时间确实在前面的示例中起作用，例如，排除了对X的干预会影响类标签。但是，这是完全可以的，实际上，通过及时进行的过程生成统计数据集是很普遍的。例如，在模型（i）中，X和Y之间的统计依赖性的根本原因是一个动态过程。作者阅读标签并计划一个运动，需要在大脑中进行复杂的过程，最后使用肌肉和笔执行该运动。仅部分理解了此过程，但这是一个及时发生的物理，动态过程，其最终结果导致X和Y的联合分布不平凡。当我们进行统计学习时，我们只关心最终结果。因此，通过及时发生的过程，不仅会产生因果结构，而且还会出现纯粹的概率结构-实际上，人们可以认为，这最终是它们发生的唯一途径。但是，在两种情况下，忽略时间通常都是有益的。在统计中，讨论统计依赖之类的概念通常不需要时间。在因果模型中，通常无需花费时间来讨论干预措施的效果。但是，这两个级别的描述都可以看作是更精确地描述现实的基本物理模型的抽象；参见表1.1。此外，请注意，模型中的变量可能不一定引用定义明确的时间实例。例如，如果心理学家调查了学生动机和表现之间的统计或因果关系，则无法轻易将这两个变量分配给特定的时间实例。引用定义明确的时间实例的测量对于诸如物理学和化学之类的“硬”科学而言是相当典型的。

### 1.4.2    Gene Perturbation

We have seen in Section 1.4.1 that different causal structures lead to different intervention  distributions.   Sometimes,  we  are  indeed  interested  in  predicting  the outcome of a random variable under such an intervention. Consider the following, in some ways over simplified, example from genetics.  Assume that we are given activity data from gene A and, correspondingly, measurements of a phenotype; see Figure 1.4 (top left) for a toy data set.  Clearly, both variables are strongly correlated. This correlation can be exploited for classical prediction: If we observe that the activity of gene A lies around 6, we expect the phenotype to lie between 12 and 16 with high probability. Similarly, for a gene B (bottom left). On the other hand,we may also be interested in predicting the phenotype afterdeletinggene A, thatis, after setting its activity to 0.6Without any knowledge of the causal structure,however, it is impossible to provide a non-trivial answer.  If gene A has a causalinfluence on the phenotype, we expect to see a drastic change after the intervention(see top right).  In fact, we may still be able to use the same linear model that wehave learned from the observational data. If, alternatively, there is a common cause,possibly a third gene C, influencing both the activity of gene B and the phenotype,the intervention on gene B will have no effect on the phenotype (see bottom right).

> 在第1.4.1节中我们已经看到，不同的因果结构导致了不同的干预分布。有时，我们确实有兴趣在这种干预下预测随机变量的结果。在某些方面，以简化形式考虑遗传学的示例。假设我们获得了来自基因A的活性数据，并相应地获得了表型的测量值；有关玩具数据集，请参见图1.4（左上）。显然，这两个变量是高度相关的。这种相关性可用于经典预测：如果我们观察到基因A的活性在6附近，则我们期望该表型在12至16之间的可能性很高。同样，对于基因B（左下）。另一方面，我们也可能对删除基因A后的表型感兴趣，也就是说，在将其活性设置为0之后，如果不了解因果结构，就不可能提供一个简单的答案。如果基因A在表型上具有因果关系，我们预计干预后会发生巨大变化（请参见右上图）。实际上，我们仍然可以使用从观测数据中学到的线性模型。或者，如果有共同的原因（可能是第三个基因C）同时影响基因B的活性和表型，则对基因B的干预将不会对表型产生影响（请参见右下）。

As in the pattern recognition example,  the models are again chosen such thatthe joint distribution over gene A and the phenotype equals the joint distributionover gene B and the phenotype.  Therefore, there is no way of telling between thetop and bottom situation from just observational data, even if sample size goes toinfinity.  Summarizing, if we are not willing to employ concepts from causality,we have to answer “I do not know” to the question of predicting a phenotype afterdeletion of a gene.

> 如在模式识别示例中一样，再次选择模型，使得在基因A和表型上的联合分布等于在基因B和表型上的联合分布。 因此，即使样本量达到无穷大，也无法通过观测数据分辨出顶部和底部情况。 总而言之，如果我们不愿意使用因果关系的概念，那么我们就必须回答“我不知道”这一问题，以预测基因缺失后的表型。