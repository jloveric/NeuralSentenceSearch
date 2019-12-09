[![Build Status](https://travis-ci.org/jloveric/NeuralSentenceSearch.svg?branch=master)](https://travis-ci.org/jloveric/NeuralSentenceSearch)

# Neural Sentence Search
Sentence search using Tensorflow sentence embeddings that works in the browser - this is a one shot learning method.  The user provides a sentence or a list of sentences (as examples) to the model and an object
that should be returned if a searched for sentence falls into that class. This is meant to be a little like elasticsearch where the search result is actually an
object.  The approach uses tensorflow sentence embedding and k nearest neighbors to compute the nearest class (there is no neural network training).  It does not run using an external database
so it can be run in the browser - however, you may need to set up some tensorflow specific initialization depending on where you decide to run it.  The code is very simple.

The module can be used for simple search with small data sets, intent classification or the first step in a larger machine learning pipeline.  My target application is chatbots that are easy to create with small data (and one shot learning) and can run entirely in the browser.  

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

    let ans2 = await nn.search("his name is Jerry")
    console.log(ans2)

    let ans3 = await nn.search("a test this is")
    console.log(ans3)
}

b()
```
with output
```javascript
    {
      result: {
        classIndex: 2,
        label: '2',
        confidences: { '0': 0, '1': 0, '2': 1 }
      },
      object: { d: { c: 'the last class' } }
    }
```
and
```javascript
    {
      result: {
        classIndex: 1,
        label: '1',
        confidences: { '0': 0, '1': 1, '2': 0 }
      },
      object: { a: 'secondClass' }
    }

```
and
```
   {
      result: {
        classIndex: 0,
        label: '0',
        confidences: { '0': 1, '1': 0, '2': 0 }
      },
      object: 'firstClass'
    }

```

## Other

If you are interested in big data, you could instead investigate gnes (Generic Neural Elastic Search) https://gnes.ai/ which can run on a cluster.