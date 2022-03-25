// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import produce from "immer";

export type SettingsTreeFieldValue =
  | { input: "boolean"; value?: boolean }
  | { input: "color"; value?: string }
  | { input: "gradient"; value?: string }
  | { input: "number"; value?: number }
  | { input: "select"; value?: string; options: string[] }
  | { input: "string"; value?: string }
  | { input: "toggle"; value?: string; options: string[] };

export type SettingsTreeField = SettingsTreeFieldValue & {
  label: string;
  placeholder?: string;
};

export type SettingsTreeFields = Record<string, SettingsTreeField>;
export type SettingsTreeChildren = Record<string, SettingsTreeNode>;

export type SettingsTreeNode = {
  label?: string;
  fields?: SettingsTreeFields;
  children?: SettingsTreeChildren;
  icon?: JSX.Element;
};

export type SettingsTree = {
  format: "settings-tree";
  showFilter?: boolean;
} & SettingsTreeNode;

export function updateSettingsTree(
  previous: SettingsTree,
  path: string[],
  value: unknown,
): SettingsTree {
  return produce(previous, (draft) => {
    let node: undefined | Partial<SettingsTreeNode> = draft;
    while (node != undefined && path.length > 1) {
      const key = path.shift()!;
      node = node.children?.[key];
    }
    const key = path.shift()!;
    const field = node?.fields?.[key];
    if (field != undefined) {
      field.value = value as SettingsTreeFieldValue["value"];
    }
  });
}
