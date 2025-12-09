import React from "react";
import { View, Text, Button, Alert } from "react-native";
import { NativeCollectors } from "@services/nativeCollectors";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { RootStackParamList } from "../App";

export type PermissionsOnboardingProps = NativeStackScreenProps<RootStackParamList, "PermissionsOnboarding">;

const PermissionsOnboardingScreen: React.FC<PermissionsOnboardingProps> = ({ navigation }) => {
  const handleGrant = async () => {
    try {
      const ok = await NativeCollectors.requestAllPermissions();
      if (ok) {
        navigation.replace("Home");
      } else {
        Alert.alert("Permissions required", "Please enable the requested permissions in system settings.");
      }
    } catch (e) {
      Alert.alert("Error", "Failed to request permissions. Please try again.");
    }
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
