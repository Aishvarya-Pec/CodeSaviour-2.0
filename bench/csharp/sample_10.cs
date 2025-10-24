// EXPECTED_ISSUES: 7

using System;
using System.IO;
class Broken9 {
    static void Main() {
        string s = null;
        Console.WriteLine(s.Length); // NullReferenceException
        int[] a = new int[2];
        int x = a[5]; // IndexOutOfRangeException
        int z = 1/0; // DivideByZeroException
        object o = 5;
        string t = (string)o; // InvalidCastException
        int n = int.Parse("abc"); // FormatException
        File.ReadAllText("missing.txt"); // FileNotFoundException
    }
}
