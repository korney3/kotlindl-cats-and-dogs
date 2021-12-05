import org.jetbrains.kotlin.gradle.tasks.KotlinCompile





plugins {
    kotlin("jvm") version "1.5.31"
}

group = "me.alisa"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation ("org.jetbrains.kotlinx:kotlin-deeplearning-api:[0.3.0]")
}

tasks.test {
    useJUnit()
}

tasks.withType<KotlinCompile>() {
    kotlinOptions.jvmTarget = "1.8"
}