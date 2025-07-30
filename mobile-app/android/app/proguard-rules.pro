# Keep TensorFlow Lite classes
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.delegates.** { *; }

# Don't warn about missing TensorFlow Lite GPU classes
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options
-dontwarn org.tensorflow.lite.gpu.**
-dontwarn org.tensorflow.lite.**

# Keep TensorFlow Lite Flutter plugin classes
-keep class io.flutter.plugins.tflite.** { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep annotations
-keepattributes *Annotation*

# Keep classes that might be accessed via reflection
-keep public class * extends android.app.Activity
-keep public class * extends android.app.Application
-keep public class * extends android.app.Service
-keep public class * extends android.content.BroadcastReceiver
-keep public class * extends android.content.ContentProvider

# Keep Parcelable implementations
-keep class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator *;
}

# Keep Flutter and Dart classes
-keep class io.flutter.** { *; }
-keep class androidx.** { *; }
-dontwarn androidx.**
