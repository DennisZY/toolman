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

