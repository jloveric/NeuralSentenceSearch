"use strict";
let use = require('@tensorflow-models/universal-sentence-encoder')
let knnClassifier = require('@tensorflow-models/knn-classifier/dist/knn-classifier.js')
let debug = require('debug')('NeuralSentenceSearch')
let tf = require('@tensorflow/tfjs-core')

class NeuralSentenceSearch {
  constructor() {
    this.model = null
    this.classifier = knnClassifier.create()
    this.dictionary = []
    this.count = 0
    this.nearestNeighbors = 1
    this.embeddingList = []
  }

  async initialize() {
    this.model = await use.load()
  }

  /**
   * Add a bunch of sentences that should be matched to the given object.  The "class" of the object
   * is generated automatically as the last index of the dictionary.
   * @param {*} obj is a single object which should be returned if any of the "sentences" is matched.
   * @param {*} sentences is a list of sentences that should be mapped to the object.
   */
  async addClass(obj, sentences) {
    this.dictionary.push(obj)

    let theseSentences = (sentences instanceof Array) ? sentences : [sentences]
    let embeddings = await this.model.embed(theseSentences)
    
    this.embeddingList.push(embeddings)

    let c = theseSentences.map((s, i) => {
      //TODO: Oh magic numbers
      this.classifier.addExample(embeddings.slice([i, 0], [1, 512]), this.dictionary.length - 1)
    })
  }

  async search(val) {
    let example = await this.model.embed(val)
    let ans = await this.classifier.predictClass(example, this.nearestNeighbors)
    let distance = this.computeDistances(ans, example)
    return { result: ans, key: this.dictionary[ans.classIndex], distance : distance }
  }

  //This only works for every example a different class since kNN doesn't compute for us
  computeDistances(ans, example) {
    let confidences = ans.confidences;
    const keys = Object.keys(confidences)
    let distance = {}

    for (let key in confidences) {
      if (confidences[key] != 0) {
        let val = Number(key)
        //Get the first value, ok, but should measure against all actually
        let nearest = this.embeddingList[val].slice([0, 0], [1, 512])
        let temp = tf.sub(example, nearest)
        distance[key] = tf.sqrt(tf.dot(temp, temp.transpose())).dataSync()[0]
      }
    }

    return distance
  }

}

module.exports = NeuralSentenceSearch