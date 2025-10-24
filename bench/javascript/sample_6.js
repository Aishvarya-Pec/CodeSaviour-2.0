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

function f5_0(a,b) { return a-b }
function f5_1(a,b) { return a-b }
function f5_2(a,b) { return a-b }
function f5_3(a,b) { return a-b }
function f5_4(a,b) { return a-b }
function f5_5(a,b) { return a-b }
function f5_6(a,b) { return a-b }
function f5_7(a,b) { return a-b }
function f5_8(a,b) { return a-b }
function f5_9(a,b) { return a-b }