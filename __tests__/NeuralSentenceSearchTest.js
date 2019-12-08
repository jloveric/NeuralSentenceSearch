"use strict";
let NeuralSentenceSearch = require("../NeuralSentenceSearch.js")
let tf = require('@tensorflow/tfjs')
tf.setBackend('cpu')

test('Add some examples', async () => {
  let nn = new NeuralSentenceSearch()

  await nn.initialize();

  await nn.addSameClass({a : "firstClass"}, "This is a test")
  await nn.addSameClass({a : "secondClass"}, ["My name is John", "Me llamo Sarah"])
  await nn.addSameClass({d : {c : "the last class"}}, ["In the galaxy", "solar system"])

  let ans1 = await nn.search("the sun")
  expect(ans1.result.classIndex).toBe(2)
  expect(ans1.object.d.c).toBe('the last class')
  console.log('ans', ans1)

  let ans2 = await nn.search("his name is Jerry")
  expect(ans2.result.classIndex).toBe(1)
  expect(ans2.object.a).toBe('secondClass')
  console.log('ans', ans2)

});