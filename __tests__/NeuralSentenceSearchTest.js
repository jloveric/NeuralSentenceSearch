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
  expect(ans1[0].classIndex).toBe(2)
  expect(ans1[0].key.d.c).toBe('the last class')
  console.log('ans', ans1)

  let ans2 = await nn.search("his name is Hudson")
  console.log('ans', ans2)
  expect(ans2[0].classIndex).toBe(1)
  expect(ans2[0].key.a).toBe('secondClass')
  

  let ans3 = await nn.search("a test this is")
  expect(ans3[0].classIndex).toBe(0)
  expect(ans3[0].key).toBe('firstClass')
  console.log('ans', ans3)

  //Now try with a larger k
  let ans4 = await nn.search("My name is Ripley from a galaxy far away and this is a test",3)
  expect(ans4[0].classIndex).toBe(1)
  expect(ans4[1].classIndex).toBe(2)
  expect(ans4[2].classIndex).toBe(0)
  console.log('ans', ans4)

});