# YOLO Example

The repository contains code for a [PyTorch Live](https://pytorch.org/live) object detection prototype. The prototype uses the [YOLOv5s model](https://github.com/ultralytics/yolov5) for the object detection task and runs on-device. It runs on Android and iOS.

**NOTE: This example uses an unreleased version of PyTorch Live including an API that is currently under development and can change for the final release.**

## How was this project bootstrapped?

The project was bootstrapped with the following command:

```
npx torchlive-cli@nightly init YOLOExample --template react-native-template-pytorch-live@nightly
```

Unused packages were removed and `react-native` upgraded to version `0.64.3`.

# Screenshots

|Android|iOS|
| --------------------- | --------------------- |
|![Screenshot of YOLOExample on Android](./screenshots/screenshot-android.png)|![Screenshot of YOLOExample on iOS](./screenshots/screenshot-ios.png)|

# Run project in emulator or on a device

## Prerequisites

Install React Native development depencencies. Follow the instructions for [Setting up the development environment](https://reactnative.dev/docs/environment-setup) as provided on the React Native website.

## Install project dependencies

Run `yarn install` to install the project dependencies.

## Start Metro server

Start the Metro server, which is needed to build the app bundle (containing the transpiled TypeScript code in the `<PROJECT>/src` directory).

```
yarn start
```

## Android

Build the `apk` for Android and install and run on the emulator (or on a physical device if connected via USB).

```
yarn android
```

See instructions on the React Native website for how to build the app in release variant.

## iOS

Install CocoaPod dependencies

```
(cd ios && pod install)
```

Build the prototype app for iOS and run it in the simulator.

```
yarn ios
```

or use the following command to open the Xcode workspace in Xcode to build and run it.

```
xed ios/YOLOExample.xcworkspace
```

See instructions on the React Native website for how to build the app in release scheme.
