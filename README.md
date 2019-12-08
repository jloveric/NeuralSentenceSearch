# Neural Sentence Search
Sentence search using Tensorflow sentence embeddings that works in the browser - this is a one shot learning method.  The user provides a sentence or a list of sentences (as examples) to the model and an object
that should be returned if a searched for sentence falls into that class. This is meant to be a little like elasticsearch where the search result is actually an
object.  The approach uses tensorflow sentence embedding and k nearest neighbors to compute the nearest class (there is no neural network training).  It does not run using an external database
so it can be run in the browser - however, you may need to set up some tensorflow specific initialization depending on where you decide to run it.  The code is very simple.

## Using

``` javascript
let { NeuralSentenceSearch } = require("neural-sentence-search")

let b = async ()=>{
    
    let nn = new NeuralSentenceSearch()

    await nn.initialize();

    await nn.addClass({a : "firstClass"}, "This is a test")
    await nn.addClass({a : "secondClass"}, ["My name is John", "Me llamo Sarah"])
    await nn.addClass({d : {c : "the last class"}}, ["In the galaxy", "solar system"])

    let ans1 = await nn.search("the sun")
    console.log(ans1)

    let ans2 = await nn.search("his name is Jerry")
    console.log(ans2)
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