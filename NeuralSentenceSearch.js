"use strict";
let use = require('@tensorflow-models/universal-sentence-encoder')
let knnClassifier = require('@tensorflow-models/knn-classifier/dist/knn-classifier.js')
let debug = require('debug')('NeuralSentenceSearch')

class NeuralSentenceSearch {
  constructor() {
    this.model = null
    this.classifier = knnClassifier.create()
    this.dictionary = []
    this.count = 0
    this.nearestNeighbors = 1
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

    let c = theseSentences.map((s,i)=>{
      this.classifier.addExample(embeddings.slice([i, 0], [1, 512]), this.dictionary.length-1)
    })

  }

  async search(val) {
    let example = await this.model.embed(val)
    let ans = await this.classifier.predictClass(example, this.nearestNeighbors)
    return { result : ans, object : this.dictionary[ans.classIndex] }
  }
}

module.exports = NeuralSentenceSearch