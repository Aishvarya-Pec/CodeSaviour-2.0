// EXPECTED_ISSUES: 5

public class Broken2 {
    public static void main(String[] args) {
        String s = null;
        System.out.println(s.length()); // NullPointerException
        int[] arr = new int[2];
        int x = arr[5]; // ArrayIndexOutOfBoundsException
        int y = 1/0; // ArithmeticException
        Object o = new Integer(5);
        String z = (String) o; // ClassCastException
        int n = Integer.parseInt("abc"); // NumberFormatException
    }
}
