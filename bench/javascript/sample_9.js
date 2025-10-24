// EXPECTED_ISSUES: 17

function bad{i}(x y) {
console.log('Syntax error missing comma')
let a = 10;
let b = '5';
let c = a + b(); // TypeError calling non-function
let d = unknown; // ReferenceError
let arr = [1,2];
arr.length = -1; // RangeError
decodeURI('%E0%A4%A'); // URIError
}


// Promise errors
Promise.allSettled([Promise.reject('x'), Promise.reject('y')]).then(() => {
  throw new AggregateError([new Error('a'), new Error('b')], 'agg');
});

function f8_0(a,b) { return a-b }
function f8_1(a,b) { return a-b }
function f8_2(a,b) { return a-b }
function f8_3(a,b) { return a-b }
function f8_4(a,b) { return a-b }
function f8_5(a,b) { return a-b }
function f8_6(a,b) { return a-b }
function f8_7(a,b) { return a-b }
function f8_8(a,b) { return a-b }
function f8_9(a,b) { return a-b }