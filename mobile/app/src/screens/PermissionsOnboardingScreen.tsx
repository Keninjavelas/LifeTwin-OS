import React from "react";
import { View, Text, Button } from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { RootStackParamList } from "../App";

export type PermissionsOnboardingProps = NativeStackScreenProps<RootStackParamList, "PermissionsOnboarding">;

const PermissionsOnboardingScreen: React.FC<PermissionsOnboardingProps> = ({ navigation }) => {
  const handleGrant = () => {
    // TODO: call into nativeCollectors.requestAllPermissions()
    navigation.replace("Home");
  };

  return (
    <View style={{ flex: 1, padding: 16, justifyContent: "center" }}>
      <Text style={{ fontSize: 22, fontWeight: "bold", marginBottom: 16 }}>Enable Device Insights</Text>
      <Text style={{ marginBottom: 12 }}>
        LifeTwin OS needs access to usage stats, notifications, and screen events to build your personal insights. All
        data stays on your device by default.
      </Text>
      <Button title="Grant permissions" onPress={handleGrant} />
    </View>
  );
};

export default PermissionsOnboardingScreen;
