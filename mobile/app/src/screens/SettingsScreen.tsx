import React from "react";
import { View, Text, Switch } from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { RootStackParamList } from "../App";

export type SettingsProps = NativeStackScreenProps<RootStackParamList, "Settings">;

const SettingsScreen: React.FC<SettingsProps> = () => {
  const [syncEnabled, setSyncEnabled] = React.useState(true);

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 20, fontWeight: "bold", marginBottom: 12 }}>Settings</Text>
      <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 16 }}>
        <Text style={{ flex: 1 }}>Enable summary sync</Text>
        <Switch value={syncEnabled} onValueChange={setSyncEnabled} />
      </View>
      <Text style={{ marginTop: 24, fontWeight: "bold" }}>About</Text>
      <Text>LifeTwin OS MLP — local-first event logging with simple predictions.</Text>
    </View>
  );
};

export default SettingsScreen;
