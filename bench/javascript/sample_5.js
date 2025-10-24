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

function f4_0(a,b) { return a-b }
function f4_1(a,b) { return a-b }
function f4_2(a,b) { return a-b }
function f4_3(a,b) { return a-b }
function f4_4(a,b) { return a-b }
function f4_5(a,b) { return a-b }
function f4_6(a,b) { return a-b }
function f4_7(a,b) { return a-b }
function f4_8(a,b) { return a-b }
function f4_9(a,b) { return a-b }