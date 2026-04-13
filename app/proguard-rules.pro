# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in the Android SDK tools proguard configuration.

# Keep OkHttp
-dontwarn okhttp3.**
-keep class okhttp3.** { *; }
