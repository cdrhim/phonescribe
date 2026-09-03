plugins {
    id("com.android.application")
}

android {
    namespace = "com.cdrhim.phonescribe"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.cdrhim.phonescribe"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"

        buildConfigField(
            "String",
            "DEFAULT_API_BASE_URL",
            "\"https://desktop-ct23ruu.tail996bd3.ts.net\""
        )
        buildConfigField(
            "String",
            "WEB_APP_URL",
            "\"https://phonescribe.vercel.app/\""
        )
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.all {
            it.useJUnit()
        }
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
}
