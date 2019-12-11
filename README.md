[![Build Status](https://travis-ci.org/jloveric/NeuralSentenceSearch.svg?branch=master)](https://travis-ci.org/jloveric/NeuralSentenceSearch)

# Neural Sentence Search
Sentence search using Tensorflow sentence embeddings that works in the browser - this is a one shot learning method.  The user provides a sentence or a list of sentences (as examples) to the model and an object
that should be returned if a searched for sentence falls into that class.  The approach uses tensorflow sentence embedding and k nearest neighbors to compute the nearest class (there is no neural network training).  It does not run using an external database
so it can be run in the browser - however, you may need to set up some tensorflow specific initialization depending on where you decide to run it.  The code is very simple.

The module can be used for simple search with small data sets, intent classification or the first step in a larger machine learning pipeline.  My target application is chatbots that are easy to create with small data (and one shot learning) and can run entirely in the browser.  

## Installing

```bash
npm install neural-sentence-search
```

## Using

``` javascript
let { NeuralSentenceSearch } = require("neural-sentence-search")

let b = async ()=>{
    
    let nn = new NeuralSentenceSearch()

    await nn.initialize();

    await nn.addClass("firstClass", "This is a test")
    await nn.addClass({a : "secondClass"}, ["My name is John", "Me llamo Sarah"])
    await nn.addClass({d : {c : "the last class"}}, ["In the galaxy", "solar system"])

    let ans1 = await nn.search("the sun")
    console.log(ans1)

    let ans2 = await nn.search("his name is Hudson")
    console.log(ans2)

    let ans3 = await nn.search("a test this is")
    console.log(ans3)

    let ans4 = await nn.search("My name is Ripley from a galaxy far away and this is a test", 3)
  console.log('ans', ans4)
}

b()
```
with output
```javascript
    [
      {
        classIndex: 2,
        label: '2',
        distance: 0.973992109298706,
        key: { d: [Object] }
      }
    ]


```
and
```javascript
    [
      {
        classIndex: 0,
        label: '0',
        distance: 0.6502495408058167,
        key: 'firstClass'
      }
    ]

```
and
```javascript
   [
      {
        classIndex: 1,
        label: '1',
        distance: 0.8455434441566467,
        key: { a: 'secondClass' }
      },
      {
        classIndex: 2,
        label: '2',
        distance: 1.0046688318252563,
        key: { d: [Object] }
      },
      {
        classIndex: 0,
        label: '0',
        distance: 1.0175029039382935,
        key: 'firstClass'
      }
    ]
```

## Other

If you are interested in big data, you could instead investigate gnes (Generic Neural Elastic Search) https://gnes.ai/ which can run on a cluster.