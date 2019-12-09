"use strict";
let NeuralSentenceSearch = require("../NeuralSentenceSearch.js")
let tf = require('@tensorflow/tfjs')
tf.setBackend('cpu')

test('Add some examples', async () => {
  let nn = new NeuralSentenceSearch()

  await nn.initialize();

  await nn.addClass("firstClass", "This is a test")
  await nn.addClass({a : "secondClass"}, ["My name is John", "Me llamo Sarah"])
  await nn.addClass({d : {c : "the last class"}}, ["In the galaxy", "solar system"])


  let ans1 = await nn.search("the sun")
  expect(ans1.result.classIndex).toBe(2)
  expect(ans1.key.d.c).toBe('the last class')
  console.log('ans', ans1)

  let ans2 = await nn.search("his name is Jerry")
  expect(ans2.result.classIndex).toBe(1)
  expect(ans2.key.a).toBe('secondClass')
  console.log('ans', ans2)

  let ans3 = await nn.search("a test this is")
  expect(ans3.result.classIndex).toBe(0)
  expect(ans3.key).toBe('firstClass')
  console.log('ans', ans3)

});